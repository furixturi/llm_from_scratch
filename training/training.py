import torch
import math
from loss.loss_calculation import calc_loss_batch_cross_entropy, calc_loss_loader_avg
from utils.get_device import get_default_device
from utils.test_model_generation import test_model_generation
from utils.benchmark import print_memory_usage


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device=None,
    eval_freq=5,  # intermediate evaluate every eval_freq number of global steps
    eval_iter=5,  # intermediate evaluation calculates average loss over the first eval_iter batches in the data loader
    epoch_fn=None,  # function to call at the end of each epoch
    start_context="The meaning of life is",
):
    if device is None:
        device = get_default_device()

    # keep track of intermediate evaluation losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    num_tokens_seen, global_step = 0, -1

    # main training loop
    for epoch in range(num_epochs):
        if device.type == "cuda":
            # torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.reset_peak_memory_stats()
        # set model to training mode
        model.train()

        # iterate over each batch
        for input_batch, target_batch in train_loader:
            if device.type == "cuda" or device.type == "mps":
                print(f"=== Global step: {global_step} ===")
            # reset loss gradients from last batch
            optimizer.zero_grad()

            # calculate loss, backpropogate to get gradients, and update model weights
            loss = calc_loss_batch_cross_entropy(
                input_batch, target_batch, model, device
            )
            if device.type == "cuda" or device.type == "mps":
                print_memory_usage(device=device, tag="After forward pass")

            loss.backward()
            if device.type == "cuda" or device.type == "mps":
                print_memory_usage(device=device, tag="After backward pass")

            optimizer.step()
            if device.type == "cuda" or device.type == "mps":
                print_memory_usage(device=device, tag="After optimization step")

            # keep track and intermediate evaluation of average losses across multiple batches
            num_tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader_avg(
                        train_loader, model, device, eval_iter
                    )
                    val_loss = calc_loss_loader_avg(
                        val_loader, model, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(num_tokens_seen)
                    print(
                        f"Epoch: {epoch}, Global step: {global_step}, "
                        f"Tokens seen: {num_tokens_seen}, "
                        f"Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}"
                    )
                model.train()

        # call epoch function if provided
        if epoch_fn is not None:
            epoch_fn(epoch, model, train_losses, val_losses, track_tokens_seen, device)
        else:
            if device.type == "cuda" or device.type == "mps":
                print_memory_usage(device=device, tag=f"Epoch {epoch} end")
            test_model_generation(model=model, input_text=start_context)

    return train_losses, val_losses, track_tokens_seen


# LLM training with learning rate warmup, cosine decay, and gradient clipping
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    eval_freq=5,
    eval_iter=5,
    warmup_steps=None,  # lr warmup
    initial_lr=3e-5,  # 0.00003
    min_lr=1e-6,  # 0.000001
    device=None,
    epoch_fn=None,
):
    if device is None:
        device = get_default_device()

    # keep track of intermediate evaluation losses and tokens seen
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    num_tokens_seen, global_step = 0, -1

    # retrieve peak learning rate from optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # calculate total training steps
    total_steps = len(train_loader) * num_epochs

    # calculate linear warmup increment
    if warmup_steps is None:
        warmup_steps = int(0.2 * total_steps)  # default to 20% warmup steps
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # linear learning rate warmup
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            # cosine annealing after warmup
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

            # apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # store current learning rate

            # calculate loss and backpropagate
            loss = calc_loss_batch_cross_entropy(
                input_batch, target_batch, model, device
            )
            loss.backward()

            # clip gradients to avoid exploding gradients
            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # update model weights
            num_tokens_seen += input_batch.numel()

            # intermediate evaluation
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader_avg(
                        train_loader, model, device, eval_iter
                    )
                    val_loss = calc_loss_loader_avg(
                        val_loader, model, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(num_tokens_seen)
                    print(
                        f"Epoch: {epoch}, Global step: {global_step}, "
                        f"Tokens seen: {num_tokens_seen}, Train loss: {train_loss:.4f}, "
                        f"Validation loss: {val_loss:.4f}"
                    )
                model.train()

        # call epoch function if provided
        if epoch_fn is not None:
            epoch_fn(
                epoch,
                model,
                train_losses,
                val_losses,
                track_tokens_seen,
                track_lrs,
                device,
            )
    return train_losses, val_losses, track_tokens_seen, track_lrs
