"""
code source: https://github.com/geronimi73/3090_shorts/blob/main/captioning/captioning.ipynb
"""
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader

from tqdm import tqdm
from helper import (
    get_VLM,
    caption,
    batch_caption,
)
from data import (
    get_data,
    get_dataloader,
    load_imagenet_labels,
)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
DTYPE = torch.bfloat16

ds = get_data()
in_labels = load_imagenet_labels()


def main(batch_size=None):
   
    processor, model = get_VLM(model="SmolVLM2")

    # Be nice to the small VLM
    prompt_template = """The image shows a {class_name}. Please come up with a short image caption, list and describe the main objects shown in the image. Keep the caption short, one sentence only."""
   
    if batch_size:
        dataloader = get_dataloader(ds, in_labels, batch_size=batch_size)

        i = 0
        for images, class_names in tqdm(dataloader):
            # Assemble prompts by inserting the actual class name into the template
            prompts = [
                prompt_template.format(class_name=class_names[i])
                for i in range(len(images))
            ]

            # Prompt model with entire batch of images
            captions = batch_caption(processor, model, images, prompts)
            print(i, captions)

            i += 1
            if i == 1:
                break
    else:
        for i, d in enumerate(ds["train"]):
            # 1) Resize image to save VRAM
            # 2) Extract classname. d["label"] is a number (e.g. 23) -> convert to string ('vulture')
            image = T.CenterCrop(256)(T.Resize(256)(d["image"]))
            class_name = in_labels[d["label"]]

            prompt = prompt_template.format(class_name=class_name) # Insert class name into prompt
            res = caption(processor, model, image, prompt)
            print(i, res)
            if i == 16:
                break


if __name__ == '__main__':
    batch_size = 16
    # main(batch_size=batch_size)
    main()
