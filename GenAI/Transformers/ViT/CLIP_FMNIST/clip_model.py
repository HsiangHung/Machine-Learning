import torch
import torch.nn as nn
import numpy as np

from transformer import (
    ImageEncoder,
    TextEncoder,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE , f"({torch.cuda.get_device_name(DEVICE )})" if torch.cuda.is_available() else "")


class CLIP(nn.Module):
    def __init__(self,
        emb_dim,
        img_size, patch_size, n_channels,
        vocab_size,
        max_seq_length,
        vit_d_model, vit_heads, vit_layers,
        text_d_model, text_heads, text_layers,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(vit_d_model, vit_heads, vit_layers, img_size, patch_size, n_channels, emb_dim)
        self.text_encoder = TextEncoder(vocab_size, text_d_model, max_seq_length, text_layers, text_heads, emb_dim)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = DEVICE


    def forward(self, image, text, mask=None):
        I_e = self.image_encoder(image)          # image-encoding
        T_e = self.text_encoder(text, mask=mask) # text-encoding

        # scaled pairwise cosine similarities [n, n]
        # logits = (I_e @ T_e.transpose(-2,-1)) * torch.exp(self.temperature)
        logits = torch.matmul(I_e, T_e.transpose(-2,-1)) * torch.exp(self.temperature)

        # symmetric loss function
        labels = torch.arange(logits.shape[0]).to(DEVICE)

        loss_i = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)

        loss = (loss_i + loss_t) / 2

        return loss
