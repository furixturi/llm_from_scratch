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


# with top-k sampling, temprature sacling, and taking EOS token into account
def generate_tokens(
    model,
    input_idx,  # (batch_size, n_tokens)
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    output_idx = input_idx
    for _ in range(max_new_tokens):
        # crop tokens in the front if the whole sequence is longer than the context size
        idx = output_idx[:, -context_size:]  # (batch_size, context_size)
        with torch.no_grad():
            logits = model(idx)  # (batch_size, context_size, vocab_size)
        last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # top-k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(
                last_token_logits, top_k, dim=-1
            )  # (batch_size, top_k)
            min_top_k = top_logits[:, -1]  # (batch_size,)
            last_token_logits = torch.where(  # torch.where creates a new tensor from two tensors (input and other) based on a condition on the input tensor's device
                last_token_logits < min_top_k,  # condition
                torch.tensor(float("-inf")).to(
                    last_token_logits.device
                ),  # "input", when condition is True. (The new tensor will be created in its device so we need to put it to the same device as last_token_logits)
                last_token_logits,  # "other", when condition is False
            )

        # temperature scaling
        if temperature > 0.0:
            last_token_logits = (
                last_token_logits / temperature
            )  # (batch_size, vocab_size)
            probs = torch.softmax(
                last_token_logits, dim=-1
            )  # (batch_size,  vocab_size)
            next_token_idx = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            next_token_idx = torch.argmax(
                last_token_logits, dim=-1, keepdim=True
            )  # (batch_size, 1)

        # support for EOS token
        if next_token_idx == eos_id:
            break

        # concatenate the new token to the output
        output_idx = torch.cat(
            (output_idx, next_token_idx), dim=1
        )  # (batch_size, n_tokens + 1)

    return output_idx
