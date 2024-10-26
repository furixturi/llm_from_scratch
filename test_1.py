import torch
import tiktoken
from gpt.gpt_model import GPTModel
from utils.generate_text import generate_text_greedy


def test_model_generation(
    model, input_text, tokenizer_encoding="gpt2", max_new_tokens=10, context_size=1024
):
    tokenizer = tiktoken.get_encoding(tokenizer_encoding)

    input_idx = tokenizer.encode(input_text)
    input_idx_batch = torch.tensor(input_idx).unsqueeze(0)  # add batch dimension‰

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

    output_idx_batch = generate_text_greedy(
        model=model,
        input_idx_batch=input_idx_batch,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
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

    print("####### Test 1 : model generation #######")
    test_model_generation(
        model=model,
        input_text="Hi, I am a large language model",
        tokenizer_encoding="gpt2",
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
