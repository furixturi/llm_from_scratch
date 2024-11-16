import torch
import tiktoken
from utils.get_device import get_default_device
from utils.generate_tokens import generate_tokens_greedy


def test_model_generation(
    model, input_text, tokenizer_encoding="gpt2", max_new_tokens=10, context_size=1024, device=None
):
    if device is None:
        device = get_default_device()
    model.to(device)

    tokenizer = tiktoken.get_encoding(tokenizer_encoding)

    input_idx = tokenizer.encode(input_text)
    input_idx_batch = torch.tensor(input_idx).unsqueeze(0)  # add batch dimension‰
    input_idx_batch = input_idx_batch.to(device)

    print(
        f"""
        {50 * '='}
        {' ' * 22}IN
        {50 * '='}
        Input text: {input_text}
        Encoded input text: {input_idx}
        Encoded input tensor shape: {input_idx_batch.shape}
        """
    )

    output_idx_batch = generate_tokens_greedy(
        model=model,
        input_idx_batch=input_idx_batch,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        device=device,
    )
    output_idx = output_idx_batch.squeeze(0).tolist()
    output_text = tokenizer.decode(output_idx)

    print(
        f"""
        {50 * '='}
        {' ' * 22}OUT
        {50 * '='}
        Encoded output tensor shape: {output_idx_batch.shape}
        Encoded output text: {output_idx}
        Output text: {output_text}
        """
    )

    return output_text
