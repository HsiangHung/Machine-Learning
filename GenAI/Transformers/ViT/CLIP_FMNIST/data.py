import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from helper import tokenizer


class FashionMNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = load_dataset("fashion_mnist")

        self.transform = T.ToTensor()

        if train:
            self.split = "train"
        else:
            self.split = "test"

        self.captions = {0: "An image of a t-shirt/top",
                        1: "An image of trousers",
                        2: "An image of a pullover",
                        3: "An image of a dress",
                        4: "An image of a coat",
                        5: "An image of a sandal",
                        6: "An image of a shirt",
                        7: "An image of a sneaker",
                        8: "An image of a bag",
                        9: "An image of an ankle boot"}

    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self, i):
        img = self.dataset[self.split][i]["image"]
        img = self.transform(img)

        cap, mask = tokenizer(self.captions[self.dataset[self.split][i]["label"]])
        mask = mask.repeat(len(mask), 1)

        return {"image": img, "caption": cap, "mask": mask}


def get_dataloader(batch_size):

    train_set = FashionMNIST(train = True)
    test_set = FashionMNIST(train = False)

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,           # Use 8 CPU cores to prepare data
        pin_memory=True,         # Fast-track data to the GPU
        prefetch_factor=2,       # Keep 2 batches ready in the queue
        persistent_workers=True  # Don't shut down workers between epochs
    )

    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,           # Use 8 CPU cores to prepare data
        pin_memory=True,         # Fast-track data to the GPU
        prefetch_factor=2,       # Keep 2 batches ready in the queue
        persistent_workers=True  # Don't shut down workers between epochs
    )

    return train_loader, test_loader
