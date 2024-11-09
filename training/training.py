import torch
from loss.loss_calculation import calc_loss_batch_cross_entropy, calc_loss_loader_avg
from utils.get_device import get_default_device


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
):
    if device is None:
        device = get_default_device()

    # keep track of intermediate evaluation losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    num_tokens_seen, global_step = 0, -1

    # main training loop
    for epoch in range(num_epochs):
        # set model to training mode
        model.train()

        # iterate over each batch
        for input_batch, target_batch in train_loader:
            # reset loss gradients from last batch
            optimizer.zero_grad()

            # calculate loss, backpropogate to get gradients, and update model weights
            loss = calc_loss_batch_cross_entropy(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()

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

    return train_losses, val_losses, track_tokens_seen
