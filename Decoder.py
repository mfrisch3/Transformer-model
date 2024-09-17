import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)