"""
Code contribution from Matt Nguyen:
1. https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
2. https://colab.research.google.com/drive/1rabTm93y39FNbu-21tDhlvYh2gp8edVR?usp=sharing
"""
import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np

class PatchEmbedding(nn.Module):
    """
    * B: Batch Size
    * C: Image Channels
    * H: Image Height
    * W: Image Width
    * P_col: Patch Column
    * P_row: Patch Row
    """
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model # Dimensionality of Model
        self.img_size = img_size # Image Size
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels

        self.linear_project = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        img = self.linear_project(img) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        img = img.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
        img = img.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
        return img


class PositionalEncoding(nn.Module):
    """
    For example, img_size=32, patch_siz=8, a 32x32 image can be broken down into 16 patches of 8x8 size. 
    In this max_seq_length would need to be 16+1=17 to create sufficient position embeddings, one for each patch, 
    and one for the class token.
    * d_model is input embedding dimension 
    """
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token

        # Creating positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch, x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe

        return x


class Attention(nn.Module):
    """
    * d_model is the input embedding dimension of each token
    * head_dim is the output dimension of q, k, v for each token
    """
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.head_dim = head_dim

        self.W_q = nn.Linear(d_model, head_dim)
        self.W_k = nn.Linear(d_model, head_dim)
        self.W_v = nn.Linear(d_model, head_dim)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Dot Product of Queries and Keys
        # attention = Q @ K.transpose(-2,-1)

        # Scaling
        # attention = attention / (self.head_size ** 0.5)
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # normalize by head_dim
        attention = torch.softmax(attention, dim=-1)

        # attention = attention @ V
        attention = torch.matmul(attention, v)

        return attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention is just running multiple heads of self-attention in parallel and combining them. 
    We can do this by adding the attention heads into a module list.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([Attention(d_model, self.head_dim) for _ in range(n_heads)])

    def forward(self, x):
        # concatenating results from all attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out


class TransformerEncoder(nn.Module):
    """
    The transformer encoder is made up of two sub-layers: 
        1. the first sub-layer performs multi-head attention
        2. the second sub-layer contains a multi-layer perceptron.
    """
    def __init__(self, d_model, n_heads, r_mlp=4):
        """
        Multilayer Perceptron (MLP): is a foundational feed-forward artificial neural network consisting of at least 
        three layers—input, hidden, and output—with fully connected neurons that use non-linear activation functions 
        to model complex, non-linear relationships.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.layer_norm_1 = nn.LayerNorm(d_model)       # Sub-Layer 1 Normalization
        self.mha = MultiHeadAttention(d_model, n_heads) # Multi-Head Attention
        self.layer_norm_2 = nn.LayerNorm(d_model)       # Sub-Layer 2 Normalization

        # Multilayer Perception: GELU is used instead of RELU because it doesn’t have RELU’s limitation of 
        # being non-differentiable at zero.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )

    def forward(self, x):
        """
        Layer normalization is an optimization technique that normalizes each input in the batch independently across 
        its features.
        """
        out = x + self.mha(self.layer_norm_1(x))      # Residual Connection After Sub-Layer 1
        out = out + self.mlp(self.layer_norm_2(out))  # Residual Connection After Sub-Layer 2
        return out
