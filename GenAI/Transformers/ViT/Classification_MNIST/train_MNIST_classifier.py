"""
This code is contributed from:
    * Building a Vision Transformer Model From Scratch:
      (https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6)
    * cocde: https://colab.research.google.com/drive/1rabTm93y39FNbu-21tDhlvYh2gp8edVR?usp=sharing#scrollTo=39_bAtOyPoIA

Build classifier model using MNIST dataset with vision transformer.

The ViT classification model training performance (lr=0.001):
    |--------|---------|---------|
    | epochs | accuracy|  loss   |
    |--------|---------|---------|
    |    5   |   92%   |  1.555  |
    |--------|---------|---------|
    |   10   |   93%   |  1.529  |
    |--------|---------|---------|
    |   50   |   95%   |  1.503  |
    |--------|---------|---------|
    |  100   |   96%   |  1.488  |
    |--------|---------|---------|
    
"""
import torch
import torch.nn as nn

from transformer import VisionTransformer
from data import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE, f"({torch.cuda.get_device_name(DEVICE)})" if torch.cuda.is_available() else "")


def test(test_loader, transformer):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = transformer(images)

            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print(f'\nModel Accuracy: {100 * correct // total} %')


def main(batch_size, n_classes, d_model, n_heads, n_layers, img_size, patch_size, n_channels, epochs=1, lr=0.001):

    train_loader, test_loader = get_dataloader(batch_size, img_size)

    print(f"training data size: {len(train_loader)}")

    transformer = VisionTransformer(n_classes, d_model, n_heads, n_layers, img_size, patch_size, n_channels).to(DEVICE)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        training_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = transformer(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.3f}')

    test(test_loader, transformer)


if __name__ == '__main__':
    n_classes = 10 # for mnist datasets, 10 classes: [0, 1, ..., 9]
    batch_size = 128

    d_model = 9
    n_heads = 3
    n_layers = 3

    img_size = (32, 32)
    patch_size = (16, 16)
    n_channels = 1

    # deep learning param
    epochs = 100
    lr = 0.001

    main(batch_size, n_classes, d_model, n_heads, n_layers, img_size, patch_size, n_channels, epochs=epochs, lr=lr)
