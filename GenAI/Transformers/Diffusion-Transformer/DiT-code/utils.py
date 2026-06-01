import torch
import torchvision.transforms as T
from datasets import load_dataset

from PIL import Image
import math
import random

from torch.utils.data import RandomSampler, DataLoader

dtype = torch.bfloat16
# CUDA device highly recommended
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


def add_random_noise(latents, timesteps=1000):
    # batch size
    bs = latents.size(0)
    # gaussian noise
    noise = torch.randn_like(latents)

    # normal distributed sigmas
    sigmas = torch.randn((bs,)).sigmoid().to(latents.device)
    timesteps = (sigmas * timesteps).to(latents.device)   # yes, `timesteps = sigmas * 1000`, let's keep it simple
    sigmas = sigmas.view([latents.size(0), *([1] * len(latents.shape[1:]))])
    latents_noisy = (1 - sigmas) * latents + sigmas * noise # (1-noise_level) * latent + noise_level * noise

    return latents_noisy.to(latents.dtype), timesteps, noise


def encode_prompt(prompt, tokenizer, text_encoder, max_length=50, add_special_tokens=False):
    # prompt = prompt.lower().strip()
    
    tokenizer.padding_side = "right"
    if isinstance(prompt, list):
        prompt = [p.lower().strip() for p in prompt]
    elif isinstance(prompt, str):
        prompt = prompt.lower().strip()
    else:
        raise Exception(f"Unknown prompt type {type(prompt)}")
        
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", 
                       max_length=max_length, truncation=True, add_special_tokens=add_special_tokens).to(device)

    # print(inputs)
    
    with torch.no_grad():
        outputs = text_encoder(**inputs)    
        # 1. Start with raw Gemma output (Shape: [1, 300, 2304])
        prompt_embeds = outputs.last_hidden_state
            
    return prompt_embeds, inputs.attention_mask


def generate(prompt, transformer, tokenizer, text_encoder, dcae, num_steps = 10, latent_dim = [1, 32, 8, 8], guidance_scale = None, neg_prompt = "", seed=None, max_prompt_tok=50, add_special_tokens=False):
    device, dtype = transformer.device, transformer.dtype
    do_cfg = guidance_scale is not None

    # Encode the prompt, +neg. prompt if classifier free guidance (CFG)
    prompt_encoded, prompt_atnmask = encode_prompt(
        [prompt, neg_prompt] if do_cfg else prompt, 
        tokenizer, 
        text_encoder,
        max_length = max_prompt_tok,
        add_special_tokens = add_special_tokens
    )
        
    # Divide 1000 -> 0 in equally sized steps
    timesteps = torch.linspace(1000, 0, num_steps + 1, device=device, dtype=dtype)
    
    # Noise level. 1.0 -> 0.0 in equally sized steps
    sigmas = timesteps / 1000
    
    latent = torch.randn(
        latent_dim, 
        generator=torch.manual_seed(seed) if seed else None
    ).to(dtype).to(device)
    
    for t, sigma_prev, sigma_next, steps_left in zip(
        timesteps, 
        sigmas[:-1], 
        sigmas[1:], 
        range(num_steps, 0, -1)
    ):
        t = t[None].to(device)

        # DiT predicts noise
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states = torch.cat([latent] * 2) if do_cfg else latent,
                timestep = torch.cat([t] * 2) if do_cfg else t,
                encoder_hidden_states=prompt_encoded,
                encoder_attention_mask=prompt_atnmask,
                return_dict=False
            )[0]

        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Remove noise from latent
        latent = latent + (sigma_next - sigma_prev) * noise_pred 

    return latent_to_PIL(latent / dcae.config["scaling_factor"], dcae)


class ImageNetARDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, splits, bs, label_dropout=None, ddp=False, col_id="image_id", col_label="label", col_latent="latent"
    ):
        self.hf_dataset = hf_dataset
        self.bs = bs
        # each split is one aspect ratio
        self.splits = splits  
        self.col_label, self.col_latent, self.col_id = col_label, col_latent, col_id
        self.label_dropout = label_dropout

        # load md2, qwen2 and smolvlm captions
        self.in1k_recaps = load_imagenet_1k_vl_enriched_recaped()

        seed = 42

        # Create a dataloader for each split (=aspect ratio)
        self.dataloaders = {}
        self.samplers = {}
        for split in splits:
            if ddp: 
                self.samplers[split] = DistributedSampler(hf_dataset[split], shuffle=True, seed=seed)
            else: 
                self.samplers[split] = RandomSampler(hf_dataset[split], generator=torch.manual_seed(seed))
            self.dataloaders[split] = DataLoader(
                hf_dataset[split], sampler=self.samplers[split], collate_fn=self.collate, batch_size=bs, num_workers=4, prefetch_factor=2
            )

    def collate(self, items):
        labels = [
            # random pick between md2, qwen2 and smolvlm
            self.in1k_recaps[i[self.col_id]][random.randint(0, 2)]
            for i in items
        ]

        # drop 10% of the labels
        if self.label_dropout:
            labels = [ label if random.random() > self.label_dropout else "" for label in labels ]

        # latents shape [B, 1, 32, W, H] -> squeeze [B, 32, W, H]
        latents = torch.Tensor([i[self.col_latent] for i in items]).squeeze()

        return labels, latents
  
    def __iter__(self):
        # Reset iterators at the beginning of each epoch
        iterators = { split: iter(dataloader) for split, dataloader in self.dataloaders.items() }
        active_dataloaders = set(self.splits)  # Track exhausted dataloaders
        current_split_index = -1
        
        while active_dataloaders:
            # Round robin: change split on every iteration (=after every batch OR after we unsucc. tried to get a batch) 
            current_split_index = (current_split_index + 1) % len(self.splits)
            split = self.splits[current_split_index]

            # Skip if this dataloader is exhausted
            if split not in active_dataloaders: continue
            
            # Try to get the next batch
            try:
                labels, latents = next(iterators[split]) 

                yield labels, latents
            # dataloader is exhausted
            except StopIteration: active_dataloaders.remove(split)

    def set_epoch(self, epoch):
        for split in self.splits: self.samplers[split].set_epoch(epoch)

    def __len__(self):
        return sum([len(self.samplers[split]) for split in self.splits]) // self.bs


def load_imagenet_1k_vl_enriched_recaped():
    import requests, gzip, json
    from io import BytesIO
    
    # URL of the gzipped JSON file
    url = "https://huggingface.co/datasets/g-ronimo/imagenet-1k-vl-enriched-recaped/resolve/main/captions.json.gz"
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
        data = json.loads(gz.read().decode('utf-8'))
    return data


def make_grid(images, rows, cols, height):
    # Check if we have enough images to fill the grid
    if len(images) > rows * cols:
        print(f"Warning: Only using the first {rows * cols} images out of {len(images)}")
        images = images[:rows * cols]
    
    # Resize all images to the specified height while maintaining aspect ratio
    resized_images = []
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(height * aspect_ratio)
        resized_images.append(img.resize((new_width, height), Image.LANCZOS))
    
    # Calculate total width required for each row and the grid height
    row_widths = []
    grid_height = 0
    
    for row in range(rows):
        start_idx = row * cols
        end_idx = min(start_idx + cols, len(resized_images))
        if start_idx >= len(resized_images):
            break
            
        # Sum the widths of all images in this row
        row_width = sum(img.width for img in resized_images[start_idx:end_idx])
        row_widths.append(row_width)
        grid_height += height
    
    # Find the maximum row width to determine grid width
    max_row_width = max(row_widths) if row_widths else 0

    # Create a new blank image for the grid
    grid_image = Image.new('RGB', (max_row_width, grid_height), color='white')
    
    # Paste images into the grid with even spacing across each row
    img_index = 0
    y_offset = 0
    
    for row in range(rows):
        start_idx = row * cols
        end_idx = min(start_idx + cols, len(resized_images))
        if start_idx >= len(resized_images):
            break
            
        # Get images for this row
        row_images = resized_images[start_idx:end_idx]
        num_images = len(row_images)
        
        # Calculate total width of images in this row
        total_img_width = sum(img.width for img in row_images)
        
        # Calculate spacing between images (if more than one image in the row)
        if num_images > 1:
            spacing = (max_row_width - total_img_width) / (num_images - 1)
        else:
            # Center a single image in the row
            spacing = 0
            
        # Place images with calculated spacing
        x_offset = 0
        for i, img in enumerate(row_images):
            grid_image.paste(img, (int(x_offset), y_offset))
            x_offset += img.width + spacing
            
        y_offset += height
    
    return grid_image


def PIL_to_latent(images, ae):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda x: x.to(dtype=ae.dtype)
    ])

    if not isinstance(images, (list, tuple)): images = [images]
    
    images_tensors = torch.cat([transform(image)[None] for image in images])
    
    with torch.no_grad():
        latent = ae.encode(images_tensors.to(ae.device))
    return latent.latent


def latent_to_PIL(latent, ae):
    with torch.no_grad():
        image_out = ae.decode(latent).sample.to("cpu")
    
    if image_out.size(0) == 1:
        # Single image processing
        image_out = torch.clamp_(image_out[0,:], -1, 1)
        image_out = image_out * 0.5 + 0.5
        return T.ToPILImage()(image_out.float())
    else:
        images = []
        for img in image_out:
            img = torch.clamp_(img, -1, 1)
            img = img * 0.5 + 0.5
            images.append(T.ToPILImage()(img.float()))
        return images


def load_IN1k256px_AR(batch_size=512, batch_size_eval=256, label_dropout=0.1):
    splits_train = ["train_AR_1_to_1", "train_AR_3_to_4", "train_AR_4_to_3"]
    splits_eval = ["validation_AR_1_to_1", "validation_AR_3_to_4", "validation_AR_4_to_3"]

    ds = load_dataset("g-ronimo/IN1k256-AR-buckets-bfl16latents_dc-ae-f32c32-sana-1.0")

    dataloader_train = ImageNetARDataset(
        ds, 
        splits=splits_train, 
        bs=batch_size, 
        label_dropout=label_dropout,
        ddp=False,
    )

    dataloader_eval = ImageNetARDataset(
        ds, 
        splits=splits_eval, 
        bs=batch_size, 
        label_dropout=None,
        ddp=False
    )

    return dataloader_train, dataloader_eval


if __name__ == '__main__':
    # load g-ronimo/IN1k256-AR-buckets-bfl16latents_dc-ae-f32c32-sana-1.0
    # Drop 10% of labels for CFG
    dataloader_train, dataloader_eval = load_IN1k256px_AR(batch_size=256, label_dropout=0.1)
