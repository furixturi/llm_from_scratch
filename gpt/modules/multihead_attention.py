import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, droupout, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be must divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads  # dimension of each head

        # Query, key, value weight matrices
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)

        # Causal attention mask to prevent attending to future tokens (mask out the upper right triangle of the attention matrix)
        ## using buffer ensures that the mask will automatically be moved to the appropriate device (CPU or GPU) during training with the model and data
        ## torch.triu() returns the upper trianglular part (on and above the diagonal) of a matrix or batch of matrices, setting the other elements to 0. diagonal=1 exclueds the diagonal itself
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # Apply an additional dropout mask to reduce overfitting
        self.dropout = nn.Dropout(droupout)

        # an optional linear projection at the end
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape

        keys = self.W_k(x)  # shape: (batch_size, seq_len, d_out)
        queries = self.W_q(x)
        values = self.W_v(x)

        # Split the keys, queries, and values into multiple heads
        ## by rolling out the last dimension (batch_size, seq_len, d_out) -> (batch_size, seq_len, n_heads, head_dim)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, seq_len, self.n_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose the dimensions to perform attention
        ## (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (self-attention)
        attn_scores = queries @ keys.transpose(
            2, 3
        )  # transpose keys last two dimensions (seq_len, head_dim) -> (head_dim, seq_len) to do dot product for each head

        # Make a bookean mask from the original mask truncated to the seq_len of this batch
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        # Apply the mask to the attention scores, fill the 1s with -inf to zero out the scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Use softmax to calculate attention weights from scaled attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # Apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # Calculate context vector applying attention weights to values
        context_vec = attn_weights @ values
        # transpose back n_heads and seq_len dimensions to prepare for head concatenation
        ## (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim
        context_vec = context_vec.transpose(1, 2)
        # concatenate the heads
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.d_out)
        # optional linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec
