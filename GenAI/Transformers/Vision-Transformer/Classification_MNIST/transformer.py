import torch
import torch.nn as nn

from helper import (
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
)


class VisionTransformer(nn.Module):
    def __init__(self, n_classes, d_model, n_heads, n_layers, img_size, patch_size, n_channels):
        super().__init__()

        self.n_classes = n_classes # Number of classes

        self.d_model = d_model # embedding dimension of model
        self.n_heads = n_heads # Number of attention heads

        # check the input images can be split evenly into patches of size patch_size and the dimensionality of the model 
        # is divisible by the number of attention heads:
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.img_size = img_size     # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(d_model, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        """
        In the forward method: 
            1. input images are first passed through the patch embeddings layer to split the image into patches and 
               get the sequence of linear embeddings for those patches. 
            2. They are then passed through the positional encoding layer to add the classification token and positional 
               encoding before being passed through the encoder modules. 
            3. The classification tokens are then passed through the classification MLP to determine the classes of the images.
        """
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        pred = self.classifier(x[:,0])
        return pred
