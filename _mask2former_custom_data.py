import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from transformers import MaskFormerImageProcessor
from data_preprocess.mask_former_dataset.dataset_path import TRAIN_PATH_MASK, TEST_PATH_MASK, YAML_PATH
import yaml

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# Assuming you have set up these paths
TRAIN_IMAGE_DIR = os.path.join(TRAIN_PATH_MASK, 'images')
TRAIN_MASK_DIR = os.path.join(TRAIN_PATH_MASK, 'masks')

TEST_IMAGE_DIR = os.path.join(TEST_PATH_MASK, 'images')
TEST_MASK_DIR = os.path.join(TEST_PATH_MASK, 'masks')

class CustomImageSegmentationDataset(Dataset):
    """Image segmentation dataset for custom directory structure."""

    def __init__(self, image_dir, mask_dir, processor, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            processor: Processor to prepare inputs for model.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform
        self.images = sorted([img for img in os.listdir(image_dir) if img.endswith('.jpg')])
        self.masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith('_mask.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.array(Image.open(image_path).convert("RGB"))
        seg = np.array(Image.open(mask_path))

        instance_seg = seg[:, :, 1]  # Green channel for instance segmentation
        class_id_map = seg[:, :, 0]  # Red channel for class segmentation
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2, 0, 1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
            inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

def create_train_transform():
    train_transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    return train_transform

def get_data_loader(train_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    return train_dataloader

def get_id2label():
    # Load class names from a YAML file
    with open(YAML_PATH, 'r') as file:
        data = yaml.safe_load(file)
        id2label = {i: label for i, label in enumerate(data['names'])}
    # Print id2label to see how it looks
    print("id2label mapping:", id2label)
    return id2label

# Define a function to unnormalize and display images
def show_image_and_mask(image_tensor, mask_tensor, label, id2label):
    # Unnormalize image
    image = (image_tensor.numpy() * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
    image = (image * 255).astype(np.uint8)
    image = np.moveaxis(image, 0, -1)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1] for displaying

    # Display the image
    plt.imshow(image)
    plt.title(f"Image with label: {id2label[label.item()]}")
    plt.axis('off')
    plt.show()

    # Process and display mask
    mask = (mask_tensor.bool().numpy() * 255).astype(np.uint8)
    mask = mask.astype(np.float32) / 255.0  # Normalize to [0,1] for displaying
    plt.imshow(mask, cmap='gray')
    plt.title(f"Mask for label: {id2label[label.item()]}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # note that you can include more fancy data augmentation methods here
    train_transform = create_train_transform()

    processor = MaskFormerImageProcessor(do_reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False,
                                         do_normalize=False)

    train_dataset = CustomImageSegmentationDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, processor=processor,
                                                   transform=train_transform)

    train_dataloader = get_data_loader(train_dataset)

    # # retrieve an image from dataloader
    # batch = next(iter(train_dataloader))
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(k, v.shape)
    #     else:
    #         print(k, len(v))
    # batch_index = 1
    #
    # unnormalized_image = (batch["pixel_values"][batch_index].numpy() * np.array(ADE_STD)[:, None, None]) + np.array(
    #     ADE_MEAN)[:, None, None]
    # unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    # unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    #
    # # Display the image
    # unnormalized_image = unnormalized_image.astype(np.float32) / 255.0  # Only do this if your data is not in [0, 1]
    # plt.imshow(unnormalized_image)
    # plt.axis('off')  # Optional: Remove axes for a cleaner image
    # plt.show()

    # Get the batch
    batch = next(iter(train_dataloader))
    batch_index = 1  # Select the pair index

    id2label = get_id2label()

    # Display image and mask
    show_image_and_mask(batch["pixel_values"][batch_index], batch["mask_labels"][batch_index][0],
                        batch["class_labels"][batch_index][0], id2label)