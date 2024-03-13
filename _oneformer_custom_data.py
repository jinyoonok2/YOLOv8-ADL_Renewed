from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import torch
from data_preprocess.one_former_mask.dataset_path import TRAIN_PATH_MASK, TRAIN_PATH_IMAGE, ID2LABEL, TEST_PATH_MASK, TEST_PATH_IMAGE
import matplotlib.pyplot as plt
import albumentations as A


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# Increment each key by 1
id2label = {class_id + 1: class_name for class_id, class_name in ID2LABEL.items()}

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, transform=None):
        self.processor = processor
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([img for img in os.listdir(image_dir) if img.endswith('.jpg')])
        self.masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith('_mask.png')])
        self.transform = transform
    def __getitem__(self, idx):
        # Load image and mask

        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
        mask = Image.open(mask_path)
        mask = np.array(mask)  # Convert mask to numpy array

        # Apply transformations to the image and mask
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask)  # Convert PIL image to numpy array
            image = augmented['image']
            mask = augmented['mask']

        # Use processor to convert this to a list of binary masks, labels, text inputs, and task inputs
        inputs = self.processor(images=image, segmentation_maps=mask, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

    def __len__(self):
        return len(self.images)

    def get_class_id_for_image(self, idx):
        """
        Get the class ID for the mask of a specific image.
        Assumes that each mask contains only one class apart from the background.
        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            int: Class ID of the mask.
        """
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)

        # Use np.max to get the highest value in the mask, assuming it represents the class ID
        class_id = np.max(mask_array)
        return class_id

def create_train_transform():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        A.Rotate(limit=35, p=0.5),  # Random rotation between -35 and +35 degrees, 50% chance to apply rotation
    ])
    return train_transform

def create_test_transform():
    test_transform = A.Compose([
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    return test_transform

processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)
processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
processor.image_processor.do_resize = False

train_dataset = CustomDataset(TRAIN_PATH_IMAGE, TRAIN_PATH_MASK, processor)
test_dataset = CustomDataset(TEST_PATH_IMAGE, TEST_PATH_MASK, processor)

# Create PyTorch dataloader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)

if __name__ == '__main__':
    example = train_dataset[0]
    for k, v in example.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    print(processor.tokenizer.batch_decode(example["text_inputs"]))


    batch = next(iter(train_dataloader))

    # Process and visualize the first batch
    for i in range(len(batch['pixel_values'])):
        # Image visualization
        unnormalized_image = (batch["pixel_values"][i].squeeze().numpy() * np.array(ADE_STD)[:, None, None]) + np.array(
            ADE_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

        # Prepare the plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns: image + mask channel 0 + mask channel 1

        # Plot the image
        axes[0].imshow(unnormalized_image)
        axes[0].set_title('Image')
        axes[0].axis('off')

        # Plot each mask channel
        for idx in range(2):  # Assuming there are 2 channels in the mask
            visual_mask = (batch["mask_labels"][i][idx].bool().numpy() * 255).astype(np.uint8)
            axes[idx + 1].imshow(visual_mask, cmap='gray')
            axes[idx + 1].set_title(f'Mask Channel {idx}')
            axes[idx + 1].axis('off')

        plt.show()