from _mask2former_custom_data import CustomImageSegmentationDataset, get_data_loader, create_train_transform, get_id2label, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR

from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
import torch
from tqdm.auto import tqdm

def main():
    train_transform = create_train_transform()

    processor = MaskFormerImageProcessor(do_reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False,
                                         do_normalize=False)

    train_dataset = CustomImageSegmentationDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR,
                                                   processor=processor,
                                                   transform=train_transform)

    train_dataloader = get_data_loader(train_dataset)

    # Replace the head of the pre-trained model
    # We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
    id2label = get_id2label()
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                              id2label=id2label,
                                                              ignore_mismatched_sizes=True)


    # Start Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    running_loss = 0.0
    num_samples = 0
    for epoch in range(20):
        print("Epoch:", epoch)
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print("Loss:", running_loss / num_samples)

            # Optimization
            optimizer.step()


if __name__ == '__main__':
    main()

