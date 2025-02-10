
import torch
import torch.nn as nn
def calc_loss_batch(input_batch, target_batch, model,
                    trainable_token_pos=-1, average_embeddings=False):

    model_output = model(input_batch)
    if average_embeddings:
        # Average over the sequence dimension (dim=1)
        logits = model_output.mean(dim=1)
    else:
        # Select embeddings at the specified token position
        logits = model_output[:, trainable_token_pos, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model,
                     num_batches=None, trainable_token_pos=-1,
                     average_embeddings=False):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
    # Extract the necessary fields from the batch dictionary
        input_batch = batch["input_ids"]
        target_batch = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model,
                         num_batches=None, trainable_token_pos=-1,
                         average_embeddings=False):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
    # Extract the necessary fields from the batch dictionary
        input_batch = batch["input_ids"]
        target_batch = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        if i < num_batches:
            input_batch, target_batch = input_batch, target_batch

            model_output = model(input_batch)
            if average_embeddings:
                # Average over the sequence dimension (dim=1)
                logits = model_output.mean(dim=1)
            else:
                # Select embeddings at the specified token position
                logits = model_output[:, trainable_token_pos, :]

            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, eval_iter,
                   trainable_token_pos=-1, average_embeddings=False):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
    model.train()
    return train_loss, val_loss
