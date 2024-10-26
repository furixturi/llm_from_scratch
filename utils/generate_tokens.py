import torch


# based on the generate_tokens_simple function in scratchpad2 notebook
# Greedy sampling, always take the predicted next token with the highest logit (probability)
def generate_tokens_greedy(model, input_idx_batch, max_new_tokens, context_size):
    # input_idx shape: (batch_size, n_tokens) in the current context

    # we will output the input tokens plus the generarted new tokens
    output_idx_batch = input_idx_batch
    # iterate over the number of new tokens to generate
    for _ in range(max_new_tokens):
        # in case the current tokens are longer than the model's supported context_size,
        # crop the tokens in the front and preserve tokens that fit in the model's `context_size`
        idx_batch = output_idx_batch[:, -context_size:]

        # get the model's prediction for the current context
        with torch.no_grad():
            logits = model(idx_batch)

        # predicted next token is at the last position of the logits, so we extract only the last token's logits.
        ## logits shape: (batch_size, context_size, vocab_size) -> next_token_logits shape: (batch_size, vocab_size)
        next_token_logits_batch = logits[:, -1, :]
        # to find the index of the token with the highest probability, we only need to find the index of the largest logit in the last dimension (vocab_size)
        ## keepdim=True ensures that the output has the same shape as the input, except in the dimension where we take the argmax
        next_token_idx_batch = torch.argmax(
            next_token_logits_batch, dim=-1, keepdim=True
        )  # shape: (batch_size, 1)
        # concatenate the new token to the output
        output_idx_batch = torch.cat((output_idx_batch, next_token_idx_batch), dim=1)

    return output_idx_batch
