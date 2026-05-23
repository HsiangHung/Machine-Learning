import torchvision.transforms as T

from torch.utils.data import  DataLoader
from datasets import load_dataset

from torchvision.datasets.mnist import MNIST


def get_dataloader(batch_size, img_size):

    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor()
    ])

    train_set = MNIST(
        root="~/datasets", train=True, download=True, transform=transform
    )
    test_set = MNIST(
        root="~/datasets", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader
