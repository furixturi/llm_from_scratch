import torch
from utils.get_device import get_default_device


def print_memory_usage(device=None, tag=""):
    if device is None:
        device = get_default_device()
    print(f"Device: {device}")

    # GPU
    if device.type == "cuda":
        print(
            f"{tag} - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
        )  # divide byte by 1024^3 to get GB
        print(
            f"{tag} - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB"
        )
        print(
            f"{tag} - CUDA peak memory used: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
        )
    # MPS (Apple silicon)
    elif device.type == "mps":
        print(
            f"{tag} - MPS memory allocated: {torch.mps.current_allocated_memory() / 1024 ** 3:.2f} GB"
        )
        print(
            f"{tag} - MPS driver memory allocated: {torch.mps.driver_allocated_memory() / 1024 ** 3:.2f} GB"
        )
        print(
            f"{tag} - MPS recommended max memory: {torch.mps.recommended_max_memory() / 1024 ** 3:.2f} GB"
        )

def collect_memory_usage(device=None):
    if device is None:
        device = get_default_device()
    print(f"Device: {device}")

    statistics = {
        "device": "cpu"
    }

    # Nvidia or AMD GPU
    if device.type == "cuda":
        statistics = {
            "device": "cuda",
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
            "peak_memory_used": torch.cuda.max_memory_allocated()
        }
    elif device.type == "mps":
        statistics = {
            "device": "mps",
            "memory_allocated": torch.mps.current_allocated_memory(),
            "driver_memory_allocated": torch.mps.driver_allocated_memory(),
            "recommended_max_memory": torch.mps.recommended_max_memory()
        }
    return statistics
