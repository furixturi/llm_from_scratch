import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = config["embedding_dim"]
        self.eps = 1e-5  # epsilon, a small constant to avoid division by zero
        # two learnable parameter matrices to adjust the scaling and shifting to  best suit the data it is processing
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # calculate mean of the last dimension
        var = x.var(
            dim=-1, keepdim=True, unbiased=False
        )  # calculate variance of the last dimension. unbiased=False means the variance is not calculated with the Bessel's correction (which would have devided by N-1 instead of N). This is compatible with the original GPT-2 which was implemented in TensorFlow.
        norm_x = (x - mean) / torch.sqrt(
            var + self.eps
        )  # calculate normalized version of the input tensor
        return self.scale * norm_x + self.shift
