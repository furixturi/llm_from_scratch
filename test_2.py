import os
from gpt.gpt_model import GPTModel
from gpt.gpt2_utils.download_gpt2_model import download_gpt2_model
from gpt.gpt2_utils.load_gpt2_weights import (
    load_gpt2_model_settings_and_params,
    load_gpt2_weights_into_model,
)
from test_1 import test_model_generation

model_configs = {
    "gpt2-small (124M)": {"embedding_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"embedding_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"embedding_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"embedding_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def test_load_gpt2_pretrained_weight(model_size, models_download_dir, model):
    # download or print "File already exists and is up-to-date: {destination}"
    download_gpt2_model(model_size=model_size, models_dir=models_download_dir)

    load_gpt2_weights_into_model(model, "models/124M")


if __name__ == "__main__":
    model_size = "124M"
    models_download_dir = "models"
    GPT_CONFIG_124M = GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "embedding_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "dropout_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias needs to be True to load OpenAI weights
    }
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    print("\n=== test generate text before loading weights ===")
    test_model_generation(
        model=model,
        input_text="Hi, I am a large language model",
        tokenizer_encoding="gpt2",
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("\n=== test model download ===")
    download_gpt2_model(model_size=model_size, models_dir=models_download_dir)

    print("\n=== test extract params from downloaded model ===")
    model_dir = os.path.join(models_download_dir, model_size)
    settings, params = load_gpt2_model_settings_and_params(model_dir=model_dir)
    print(f"settings: {settings}, params.keys(): {params.keys()}")

    print("\n=== test load params into model ===")
    load_gpt2_weights_into_model(model, params)
    print("No errors loading weights into model.")

    print("\n=== generate text after loading weights ===")
    test_model_generation(
        model=model,
        input_text="Hi, I am a large language model",
        tokenizer_encoding="gpt2",
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
