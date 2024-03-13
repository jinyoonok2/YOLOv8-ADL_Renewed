from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.optim import AdamW

from _oneformer_custom_data import train_dataloader, processor
import torch
import os
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Directory where you want to save and load your model checkpoints **********
    checkpoint_dir = 'one_former/train1-epoch50'
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it does not exist


    # Function to find the latest checkpoint in the directory
    def find_latest_checkpoint(checkpoint_dir):
        checkpoint_paths = [p for p in Path(checkpoint_dir).glob("checkpoint_epoch_*.pt")]
        if checkpoint_paths:
            latest_checkpoint = max(checkpoint_paths, key=os.path.getctime)
            return latest_checkpoint
        return None


    # Try to load the latest checkpoint
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0

    if latest_checkpoint_path and os.path.isfile(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("No checkpoint found, starting from scratch.")

    model.train()
    model.to(device)

    # Training loop
    num_epochs = 100
    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
        total_loss = 0  # Initialize total loss for the epoch
        num_batches = 0  # Keep track of the number of batches processed

        # Wrap your dataloader with tqdm for a progress bar
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, batch in progress_bar:
            # zero the parameter gradients
            optimizer.zero_grad()

            # transfer batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # forward pass
            outputs = model(**batch)

            # calculate loss
            loss = outputs.loss
            total_loss += loss.item()  # Aggregate the loss
            num_batches += 1  # Increment the number of batches

            # backward pass + optimize
            loss.backward()
            optimizer.step()

            # Update progress bar description with current loss
            progress_bar.set_description(f"Epoch {epoch}/{num_epochs - 1} Loss: {loss.item()}")

        avg_loss = total_loss / num_batches  # Calculate average loss for the epoch
        print(f"Epoch: {epoch}, Avg Loss: {avg_loss}")  # Print average loss for the epoch

        # Save model and optimizer states after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,  # Save the average loss for the epoch
        }, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
