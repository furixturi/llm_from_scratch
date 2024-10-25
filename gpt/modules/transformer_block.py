import torch
import torch.nn as nn

from gpt.modules.layer_norm import LayerNorm
from gpt.modules.multihead_attention import MultiHeadAttention
from gpt.modules.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config)
        self.att = MultiHeadAttention(
            d_in=config["embedding_dim"],
            d_out=config["embedding_dim"],
            context_length=config["context_length"],
            droupout=config["dropout_rate"],
            n_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )

        self.norm2 = LayerNorm(config)
        self.ff = FeedForward(config)

        self.drop_shortcut = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        # shortcut connection around multi-head attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # shortcut connection around feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
