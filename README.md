# LLM from scratch

Reference: https://github.com/rasbt/LLMs-from-scratch/tree/main

## How to run
1. (Optional) Create a Python virtual environment and activate it
```bash
$ python -m venv .venv
$ source .vent/bin/activate
```
2. Install dependencies
```bash
$ pip3 install -r requirements.txt
```
3. Run dev setup script to support local module import
```bash
$ source setup_env.sh
```


## GPT

### GPT2 Architecture

![](./assets/GPT_architecture.png)

```python

model_configs = {
    "gpt2-small (124M)": {"embedding_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"embedding_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"embedding_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"embedding_dim": 1600, "n_layers": 48, "n_heads": 25},
}
```

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Vocabulary size
    "context_length": 1024,     # Context length
    "embedding_dim": 768,             # Embedding dimension
    "n_heads": 12,              # Number of attention heads
    "n_layers": 12,             # Number of layers
    "dropout_rate": 0.1,           # Dropout rate
    "qkv_bias": False           # Query-Key-Value bias
}
```

## Development Plan

```
[x] GPT model framework
    [x] LayerNorm
    [x] Transformer block
        [x] Multihead Attention
        [x] FeedForward
[x] generate text
[ ] Save and load model
    [ ] PyTorch
    [ ] HuggingFace
[ ] Data loader
[ ] Pretrain
    [ ] Training script
[ ] Evaluation
[ ] Llama
[ ] Fine-tune
```
