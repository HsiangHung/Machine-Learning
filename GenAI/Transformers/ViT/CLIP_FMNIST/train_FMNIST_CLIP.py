"""
This code is contributed from:
    * Building CLIP From Scratch:
      (https://medium.com/correll-lab/building-clip-from-scratch-68f6e42d35f4)
    * code: https://colab.research.google.com/drive/1E4sEg7RM8HBv4PkIhjWZuwCXXbB_MinS?usp=sharing#scrollTo=SHKmUq7V_qhH

Build CLIP model using fashion MNIST dataset from HuggingFace.

CLIP model training Performance (lr=0.0001):
    | -------|--------|----------|---------|
    | epochs |   lr   | accuracy |   loss  |
    |--------|--------|----------|---------|
    |   10   | 0.0001 |    80%   |  3.056  |
    |--------|--------|----------|---------|
    |   50   | 0.0001 |    84%   |  2.657  |
    |--------|--------|----------|---------|
    |  100   | 0.0001 |    85%   |  2.743  |
    |--------|--------|----------|---------|
    |  200   | 0.0001 |    85%   |  2.668  | 
    |--------|--------|----------|---------|
    |  100   | 0.001  |    86%   |  2.585  |
    |--------|--------|----------|---------|
    |  200   | 0.001  |    85%   |  2.570  |
    |--------|--------|----------|---------|
"""
import torch
import numpy as np

from data import (
    get_dataloader,
    FashionMNIST,
)
from clip_model import CLIP
from helper import (
    tokenizer,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE , f"({torch.cuda.get_device_name(DEVICE )})" if torch.cuda.is_available() else "")


def test(test_loader, model):
    test_set = FashionMNIST(train = False)

    text = torch.stack([tokenizer(x)[0] for x in test_set.captions.values()]).to(DEVICE)
    mask = torch.stack([tokenizer(x)[1] for x in test_set.captions.values()])
    mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(DEVICE)

    correct, total = 0,0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data["image"].to(DEVICE), data["caption"].to(DEVICE)
            image_features = model.image_encoder(images)
            text_features = model.text_encoder(text, mask=mask)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            similarity = torch.matmul(100.0 * image_features, text_features.T).softmax(dim=-1)
            _, indices = torch.max(similarity, 1)

            pred = torch.stack([tokenizer(test_set.captions[int(i)])[0] for i in indices]).to(DEVICE)
            correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))
            total += len(labels)

    print(f'\nModel Accuracy: {100 * correct // total} %')


def main(
        emb_dim, 
        img_size, patch_size, n_channels, 
        vocab_size, 
        max_seq_length, 
        vit_d_model, vit_heads, vit_layers,
        text_d_model, text_heads, text_layers, 
        batch_size, epochs=1, lr=0.001,
):

    train_loader, test_loader = get_dataloader(batch_size)

    model = CLIP(
        emb_dim,
        img_size, patch_size, n_channels,
        vocab_size,
        max_seq_length,
        vit_d_model, vit_heads, vit_layers,
        text_d_model, text_heads, text_layers,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_loss = np.inf
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            img, cap, mask = data["image"].to(DEVICE), data["caption"].to(DEVICE), data["mask"].to(DEVICE)
            loss = model(img, cap, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Batch Loss: {loss.item():.3f}")

        # Saves model if it performed better than the previous best
        if loss.item() <= best_loss:
            best_loss = loss.item()
            # if epoch % 20 == 0:
                # torch.save(model.state_dict(), "./clip.pt")
                # print(f"Model Saved on {epoch}.")

    test(test_loader, model)


if __name__ == '__main__':
    emb_dim = 32 # in CLIP both image and text are described by same embedding dim.

    img_size = (28, 28)
    patch_size = (14, 14)
    n_channels = 1

    vit_layers = 3
    vit_heads = 3
    vit_d_model = 9

    max_seq_length = 32

    vocab_size = 256
    text_layers = 4
    text_heads = 8
    text_d_model = 32

    lr = 1e-3
    epochs = 200

    batch_size = 128

    main(
        emb_dim, 
        img_size, patch_size, n_channels, 
        vocab_size, 
        max_seq_length, 
        vit_d_model, vit_heads, vit_layers,
        text_d_model, text_heads, text_layers, 
        batch_size, epochs=epochs, lr=lr
    )
