from pathlib import Path
import numpy as np
import cv2
import os
from dataset_path import TRAIN_PATH_LABEL, TRAIN_PATH_MASK, TEST_PATH_LABEL, TEST_PATH_MASK

# Function to create semantic masks
def create_semantic_mask(label_file, img_shape):
    semantic_mask = np.zeros(img_shape, dtype=np.uint8)  # Initialize mask with 0 (background)

    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0]) + 1  # Increment class_id by 1 to reserve 0 for background
            points = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
            points[:, 0] *= img_shape[1]  # Denormalize x
            points[:, 1] *= img_shape[0]  # Denormalize y
            points = points.astype(np.int32)

            # Draw polygon on the mask with the class_id
            cv2.fillPoly(semantic_mask, [points], color=class_id)

    return semantic_mask

# Function to process the label files and create semantic masks
def process_labels(label_path, mask_path):
    os.makedirs(mask_path, exist_ok=True)  # Create mask path if it doesn't exist
    for label_file in Path(label_path).glob('*.txt'):
        # Read label file and create semantic mask
        img_shape = (256, 256)  # Fixed image size
        semantic_mask = create_semantic_mask(label_file, img_shape)

        # Save semantic mask to mask path
        mask_filename = os.path.join(mask_path, label_file.stem + '_mask.png')
        cv2.imwrite(mask_filename, semantic_mask)

if __name__ == '__main__':
    # Process label files for combined training and testing
    # process_labels(COMB_TRAIN_PATH_LABELS, COMB_TRAIN_PATH_MASKS)
    process_labels(TRAIN_PATH_LABEL, TRAIN_PATH_MASK)
    process_labels(TEST_PATH_LABEL, TEST_PATH_MASK)

