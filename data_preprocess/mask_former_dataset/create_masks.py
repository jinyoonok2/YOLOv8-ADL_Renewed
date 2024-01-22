import os
import cv2
import yaml
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

# Assuming you have these paths set up already
from dataset_path import TRAIN_PATH_MASK, TEST_PATH_MASK, YAML_PATH

mask_counts = defaultdict(int)


# Function to create mask for an image
def create_mask(image_path, label_path, mask_path):
    # Load the image to get its dimensions
    image = Image.open(image_path)
    width, height = image.size

    # Create an empty mask
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Read the YOLO annotation file
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Process each object in the annotation
    for line in lines:
        parts = [float(x) for x in line.strip().split()]
        class_id, normalized_vertices = int(parts[0]), parts[1:]

        # Ensure class_id is an integer within the range 0-255
        class_id = int(class_id)
        if class_id < 0 or class_id > 255:
            raise ValueError(f"class_id {class_id} out of range. It should be between 0 and 255.")

        # Convert normalized coordinates to actual pixel coordinates
        vertices = [int(v * width if i % 2 == 0 else v * height) for i, v in enumerate(normalized_vertices)]

        # Create a unique instance ID for each object.
        # Ensure instance_id is an integer within the range 1-255 (0 is usually reserved for background)
        instance_id = int(np.max(mask[:, :, 1])) + 1
        if instance_id < 1 or instance_id > 255:
            raise ValueError(f"instance_id {instance_id} out of range. It should be between 1 and 255.")

        # R(ed) channel encodes category ID, G(reen) channel encodes instance ID
        polygon = np.array(vertices).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], (class_id, instance_id, 0))

    # Save the mask
    mask_image = Image.fromarray(mask)
    mask_image.save(mask_path)
    print(f"Mask created: {mask_path}")


# Function to process directories
def process_directory(data_path, data_type):
    image_dir = os.path.join(data_path, 'images')
    label_dir = os.path.join(data_path, 'labels')
    mask_dir = os.path.join(data_path, 'masks')

    # Ensure the mask directory exists
    os.makedirs(mask_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
            mask_path = os.path.join(mask_dir, filename.replace('.jpg', '_mask.png'))

            # Create mask for each image
            if os.path.exists(label_path):
                create_mask(image_path, label_path, mask_path)
                mask_counts[data_type] += 1


# Process TRAIN and TEST directories
process_directory(TRAIN_PATH_MASK, 'TRAIN')
process_directory(TEST_PATH_MASK, 'TEST')

# Print the counts for each dataset type
for data_type, count in mask_counts.items():
    print(f"Total masks created for {data_type}: {count}")


# # Test the function with a single image and label
# image_path = os.path.join(TRAIN_PATH, 'images', 'BS_TRAIN_1.jpg')
# label_path = os.path.join(TRAIN_PATH, 'labels', 'BS_TRAIN_1.txt')
# mask_path = os.path.join(TRAIN_PATH, 'masks', 'BS_TRAIN_1_mask.png')
# create_mask(image_path, label_path, mask_path)