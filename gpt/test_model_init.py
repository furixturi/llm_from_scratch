import torch
import tiktoken
from gpt.gpt_model import GPTModel


def generate_tokens_simple(model, input_idx, max_new_tokens, context_size):
    # input_idx shape: (batch_size, n_tokens) in the current context

    # we will output the input tokens plus the generarted new tokens
    output_idx = input_idx
    # iterate over the number of new tokens to generate
    for _ in range(max_new_tokens):
        # in case the current tokens are longer than the model's supported context_size,
        # crop the tokens in the front and preserve tokens that fit in the model's `context_size`
        idx = output_idx[:, -context_size:]

        # get the model's prediction for the current context
        with torch.no_grad():
            logits = model(idx)

        # predicted next token is at the last position of the logits, so we extract only the last token's logits.
        ## logits shape: (batch_size, context_size, vocab_size) -> next_token_logits shape: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]
        # to find the index of the token with the highest probability, we only need to find the index of the largest logit in the last dimension (vocab_size)
        ## keepdim=True ensures that the output has the same shape as the input, except in the dimension where we take the argmax
        next_token_idx = torch.argmax(
            next_token_logits, dim=-1, keepdim=True
        )  # shape: (batch_size, 1)
        # concatenate the new token to the output
        output_idx = torch.cat((output_idx, next_token_idx), dim=1)

    return output_idx


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "embedding_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "dropout_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
