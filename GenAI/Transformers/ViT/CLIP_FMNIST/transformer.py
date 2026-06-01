import torch
import torch.nn as nn

from helper import (
    PositionalEmbedding,
    TransformerEncoder,
)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, n_layers, n_heads, emb_dim):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)

        self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
        self.encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])

        # learned proj of image to embed
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))

    def forward(self, token, mask=None):
        word_embedding = self.encoder_embedding(token)          # Text Embedding
        pos_encoded = self.positional_embedding(word_embedding) # Positional Embedding

        # Transformer Encoder
        for encoder_layer in self.encoder:
            pos_encoded = encoder_layer(pos_encoded, mask=mask)

        # Takes features from the EOT Embedding
        x = pos_encoded[torch.arange(token.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]

        # joint multimodal embedding
        if self.projection is not None:
            # x = x @ self.projection
            x = torch.matmul(x, self.projection)

        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, img_size, patch_size, n_channels, emb_dim):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
        self.max_seq_length = self.n_patches + 1

        self.linear_project = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.positional_embedding = PositionalEmbedding(d_model, self.max_seq_length)
        self.encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])

        # learned proj of image to embed
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))


    def forward(self, image):
        """
        * B: Batch Size
        * C: Image Channels
        * H: Image Height
        * W: Image Width
        * P_col: Patch Column
        * P_row: Patch Row
        """
        # Patch Embedding
        x = self.linear_project(image) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(2)               # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2)          # (B, d_model, P) -> (B, P, d_model)

        # Positional Embedding
        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1),x), dim=1)
        x = self.positional_embedding(x)

        # Transformer Encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # Getting Class Tokens
        x = x[:, 0, :]

        # joint multimodal embedding
        if self.projection is not None:
            # x = x @ self.projection
            x = torch.matmul(x, self.projection)

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x
