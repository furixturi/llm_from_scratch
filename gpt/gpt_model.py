import torch
import torch.nn as nn

import modules.layer_norm as LayerNorm
import modules.transformer_block as TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # embedding layers
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["embedding_dim"])
        # first dropout layer
        self.dropout_emb = nn.Dropout(config["dropout_rate"])

        # transformer blocks
        self.transformer_blocks = nn.Sequential(
            # unpacking operator "*" unpacks a list of TransformerBlock objects as arguments to nn.Sequential
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        # final LayerNorm layer
        self.final_norm = LayerNorm(config)
        # output head layer
        ## its input is the final hidden state of shape (batch_size, context_length, embedding_dim)
        ## its output is a logits vector of shape (batch_size, context_length, vocab_size), each value in the last dimension corresponds to that token ID's score in the whole vacabulary
        ## each sequence position in the output represents the predicted next token of the corresponding position's token in the input
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"])

    def forward(self, in_idx):
        batch_size, seq_len = (
            in_idx.shape
        )  # in_idx is a tensor of input token IDs of shape (batch_size, seq_len, vocab_size)
        tok_emb = self.tok_emb(
            in_idx
        )  # will return each token's token embeddings of shape (batch_size, seq_len, embedding_dim)
        pos_emb = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )  # will return each token's positional embeddings of shape (batch_size seq_len, embedding_dim)
        # add positional embeddings to token embeddings
        x = tok_emb + pos_emb

        # pass through first dropout layer
        x = self.dropout_emb(x)
        # pass through transformer blocks
        x = self.transformer_blocks(x)
        # pass through final LayerNorm layer
        x = self.final_norm(x)

        # pass through output head layer to get the logits
        logits = self.out_head(
            x
        )  # logits is of shape (batch_size, seq_len, vocab_size)
        return logits
