"""
Code contributed from section "Diffusion Transformer" of blog "Training a Latent Diffusion Model From Scratch"
(https://medium.com/@geronimo7/training-a-latent-diffusion-model-from-scratch-897c7b77ece9)

* Diffusion Transformer (DiT) plays a role as an actual noise-predicting model.

"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms as T

from datasets import load_dataset
from diffusers import SanaTransformer2DModel, AutoencoderDC
from transformers import AutoModel, AutoTokenizer, GemmaTokenizerFast

from utils import (
    add_random_noise,
    encode_prompt,
    generate,
)

# Train with reduced precision 
DTYPE = torch.bfloat16
# CUDA device highly recommended
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

print(f"device: {DEVICE}")

CHECKPOINT_DIR = "/mnt/localssd/checkpoints/imagenet_diffusion-2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def build_DiT():
    # Load Sana 600M config
    config = SanaTransformer2DModel.load_config(
        "Efficient-Large-Model/Sana_600M_1024px_diffusers", 
        subfolder="transformer"
    )

    # Reduce depth
    config["num_layers"] = 12

    # Reduce width
    config["num_attention_heads"] = 12
    config["attention_head_dim"] = 64
    config["cross_attention_dim"] = 768
    config["num_cross_attention_heads"] = 12
    config["cross_attention_head_dim"] = 64

    # config["num_layers"] = 24
    # # Reduce width
    # config["num_attention_heads"] = 15
    # config["attention_head_dim"] = 64
    # config["cross_attention_dim"] = 960
    # config["num_cross_attention_heads"] = 15
    # config["cross_attention_head_dim"] = 64

    # Adapt to hidden size of SmolLM2
    config["caption_channels"] = 960

    transformer = SanaTransformer2DModel.from_config(config).to(DTYPE).to(DEVICE)
    print("transformer is ready")
    return transformer


def build_tokenizer(repo_path="HuggingFaceTB/SmolLM2-360M"):
    # Load the text encoder and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_path, torch_dtype=DTYPE)
    # Pad token is not set by default, use eos
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer is ready")
    return tokenizer


def build_text_encoder(repo_path="HuggingFaceTB/SmolLM2-360M"):
    text_encoder = AutoModel.from_pretrained(repo_path, torch_dtype=DTYPE).to(DEVICE)
    print("text_encoder is ready")
    return text_encoder


def build_vae(repo_path="Efficient-Large-Model/Sana_600M_1024px_diffusers"):
    # Load Deep Compression AutoEncoder
    vae = AutoencoderDC.from_pretrained(repo_path, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
    print("vae is ready")
    return vae


def get_dataloader():
    from utils import load_IN1k256px_AR
    dataloader_train, dataloader_eval = load_IN1k256px_AR(batch_size=256, label_dropout=0.1)
    # dataloader_train, dataloader_eval = load_IN1k256px_AR(batch_size=128, label_dropout=0.1)
    print("dataloader is ready")
    return dataloader_train, dataloader_eval


def get_checkoutpoint(
    transformer,
    optimizer,
    checkpoint_dir=CHECKPOINT_DIR,
    checkpoint_filename="transformer_latest.pt"
):
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: {checkpoint_path}. Loading...")
        
        # 1. Load the dictionary to CPU first (safest)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 2. Load the model weights    
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        loss_history = checkpoint['loss_history']
        print(f"Successfully resumed from step {start_step}")

        transformer.to(DEVICE)
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_step = 0
    
    return transformer, loss_history


def validate_DiT(transformer, optimizer, tokenizer, text_encoder, vae):
    transformer, loss_history = get_checkoutpoint(transformer, optimizer, checkpoint_filename="transformer_latest.pt")
    generate(
        "a mountain and its reflection in a lake",
        transformer, tokenizer, text_encoder, vae,
        seed=42,
        num_steps=20,
        guidance_scale=7
    )
    print(loss_history[-5:])


def train_DiT(transformer, optimizer, tokenizer, text_encoder, vae, dataloader_train, epochs=100):
    # optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)
    transformer.train()

    # epochs = 1

    step = 0 
    loss_history = []
    for e in range(epochs):
        for text_labels, img_latents in dataloader_train:
            step += 1
            epoch = step/len(dataloader_train)

            # Encode prompts
            # text_labels is a batch of strings, e.g. ['A trifle with chocolate,.....', 'A New Home sew...', ...]
            # len(text_labels) = batch_size
            prompts_emb, prompts_atnmask = encode_prompt(text_labels, max_length=50, tokenizer=tokenizer, text_encoder=text_encoder)

            # Scale image latent and add random amount of noise, img_latents.shape=(batch_size, d1, d2, d3)
            img_latents = img_latents.to(DTYPE).to(DEVICE)
            img_latents *=  vae.config["scaling_factor"]
            latents_noisy, timestep, noise = add_random_noise(img_latents)
            
            # Get a noise prediction out of the model
            noise_pred = transformer(
                hidden_states = latents_noisy.to(DTYPE), 
                encoder_hidden_states = prompts_emb, 
                encoder_attention_mask = prompts_atnmask,
                timestep = timestep, 
            ).sample

            optimizer.zero_grad()

            # Calculate gradients
            loss = F.mse_loss(noise_pred.float(), (noise - img_latents).float())
            loss.backward()
            
            # Clip gradients
            # grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=0.1)

            # Update weights
            optimizer.step()
        
            if step % 100 == 0:
                print(f"step {step} epoch {epoch:.2f} loss: {loss.item()}")
                loss_history.append(loss.item())

            if step % len(dataloader_train) == 0:
                # store checkpoint every epoch
                checkpoint = {
                    'step': step,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'loss_history': loss_history,
                }
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"transformer_latest_epoch-{int(epoch)}.pt"))


def main(train=True, val=False):

    transformer = build_DiT()

    te_repo = "HuggingFaceTB/SmolLM2-360M"
    tokenizer, text_encoder = build_tokenizer(repo_path=te_repo), build_text_encoder(repo_path=te_repo)

    sana_repo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
    vae = build_vae(repo_path=sana_repo)

    dataloader_train, dataloader_eval = get_dataloader()

    learning_rate = 0.0005
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    if train:
        max_epochs = 200
        train_DiT(transformer, optimizer, tokenizer, text_encoder, vae, dataloader_train, epochs=max_epochs)

    if val:
        validate_DiT(transformer, optimizer, tokenizer, text_encoder, vae)


if __name__ == '__main__':
    main(train=True, val=False)
