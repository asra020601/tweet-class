from evaluate import evaluate_model, calc_accuracy_loader, calc_loss_batch
from transformer import model
from dataloader import train_loader, val_loader
from multiprocessing import freeze_support
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def train_classifier_simple(model, train_loader, val_loader, optimizer, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            average_embeddings=False):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch in train_loader:
            input_batch = batch["input_ids"]         # inputs for the model
            attention_mask = batch["attention_mask"]   # if your loss or model requires an attention mask
            target_batch = batch["labels"]             # target labels for computing loss

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model,
                                   trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen
import time
import torch

# Ensure GPU is utilized
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

start_time = time.time()


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 10  # Reduce epochs for testing
freeze_support()
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(model, train_loader, val_loader, optimizer,num_epochs=num_epochs, eval_freq=200, eval_iter=10,  # Evaluate less frequently
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, "checkpoint.pth")
model.eval()
this = model.save()