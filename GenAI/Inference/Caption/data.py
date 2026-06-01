from torch.utils.data import DataLoader
from torchvision import transforms as T
import requests, json
from datasets import load_dataset

def get_data():
    from huggingface_hub import login
    token = "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"
    return load_dataset("ILSVRC/imagenet-1k", streaming=True, trust_remote_code=True, token=token)


def load_imagenet_labels():
    raw_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(raw_url)
    imagenet_labels = json.loads(response.text)
    return imagenet_labels


def get_dataloader(ds, in_labels, batch_size=16):
    # A collate function converts a list of items to a batch
    def collate_fn(items):
        return [
            [T.Resize(256)(i["image"]) for i in items],
            [in_labels[i["label"]] for i in items],
        ]
    
    # DataLoader's job is to prepare and preload batches of data
    return DataLoader(
        ds["train"], 
        shuffle=False, 
        collate_fn=collate_fn, 
        batch_size=batch_size,     
        num_workers=8,      # Use 2 threads for prefetching
        prefetch_factor=10  # Each thread will prefetch 10 samples
    )