import torch
import torch.nn as nn

import numpy as np


class Attention(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        
        self.d_model = d_model
        self.head_dim = head_dim

        self.W_q = nn.Linear(d_model, head_dim)
        self.W_k = nn.Linear(d_model, head_dim)
        self.W_v = nn.Linear(d_model, head_dim)
        
    def forward(self, x, mask=None):
        # Obtaining Queries, Keys, and Values
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # return out

        # Dot Product of Queries and Keys
        # attention = q @ k.transpose(-2,-1)
        # attention = torch.matmul(q, k.transpose(-2,-1)) 

        # Scaling
        # attention = attention / (self.head_size ** 0.5)

        attention = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5)

        # Applying Attention Mask
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(attention, dim=-1)
        # attention = attention @ v
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
        self.multiheads = nn.ModuleList([Attention(d_model, self.head_dim) for _ in range(n_heads)])

    def forward(self, x, mask=None):
        # Combine attention heads
        out = torch.cat([head(x, mask=mask) for head in self.multiheads], dim=-1)
        out = self.W_o(out)
        return out


class TransformerEncoder(nn.Module):
    """
        Multilayer Perceptron (MLP): is a foundational feed-forward artificial neural network consisting of at least 
        three layers—input, hidden, and output—with fully connected neurons that use non-linear activation functions 
        to model complex, non-linear relationships.
        """
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.layer_norm_1 = nn.LayerNorm(d_model)        # Sub-Layer 1 Normalization
        self.mha = MultiHeadAttention(d_model, n_heads)  # Multi-Head Attention
        self.layer_norm_2 = nn.LayerNorm(d_model)        # Sub-Layer 2 Normalization

        # Multilayer Perception: GELU is used instead of RELU because it doesn’t have RELU’s limitation of 
        # being non-differentiable at zero.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )

    def forward(self, x, mask=None):
        # Residual Connection After Sub-Layer 1
        x = x + self.mha(self.layer_norm_1(x), mask=mask)

        # Residual Connection After Sub-Layer 2
        x = x + self.mlp(self.layer_norm_2(x))
        return x


def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        out = chr(2) + text + chr(3) # Adding <SOT> at the begining and <EOT> in the end
        out = out + "".join([chr(0) for _ in range(max_seq_length-len(out))]) # Adding Padding
        out = torch.IntTensor(list(out.encode("utf-8"))) # Encoding Text
        mask = torch.ones(len(out.nonzero()))
        mask = torch.cat((mask,torch.zeros(max_seq_length-len(mask)))).type(torch.IntTensor)
    else:
        out = [chr(x) for x in text[1:len(mask.nonzero())-1]]
        out = "".join(out)
        mask = None

    return out, mask


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                  pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
    
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe
        return x
