import torch
from utils.get_device import get_default_device


def calc_loss_batch_cross_entropy(input_batch, target_batch, model, device=None):
    if device is None:
        device = get_default_device()

    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader_avg(
    data_loader,
    model,
    device=None,
    num_batches=None,
    batch_loss_fn=calc_loss_batch_cross_entropy,
):
    """
    Calculate the average loss of a model on a data loader's first num_batches data batches.
    """
    if device is None:
        device = get_default_device()

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = batch_loss_fn(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
