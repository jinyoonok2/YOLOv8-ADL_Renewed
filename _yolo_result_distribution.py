import os
from glob import glob
import glob
import shutil
from sklearn.model_selection import train_test_split
from data_preprocess.yolo_txt.__dataset_path import (ACTIVE_LEARNING_PATH, INFER_PATH, ID2LABEL,
                                                     TRAIN_PATH, VALID_PATH, CLASSES)
import pandas as pd


def correct_data(model_path):
    # Extract experiment name from model_path
    experiment_name = model_path.split('/')[2]
    error_csv_path = os.path.join(INFER_PATH, 'class_id_errors.csv')

    # Initialize or load the DataFrame for error percentages
    if os.path.exists(error_csv_path):
        error_df = pd.read_csv(error_csv_path, index_col=0)
    else:
        error_df = pd.DataFrame()

    # Iterate over each class directory in INFER_PATH
    for class_id, class_name in ID2LABEL.items():
        class_dir = os.path.join(INFER_PATH, class_name)
        label_files = glob.glob(os.path.join(class_dir, "*.txt"))
        error_count = 0
        total_labels = 0

        # Correct the class_id in each label file
        for label_file in label_files:
            with open(label_file, 'r') as file:
                lines = file.readlines()

            corrected_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] != str(class_id):
                    parts[0] = str(class_id)  # Correct the class_id
                    error_count += 1
                corrected_lines.append(" ".join(parts))
            total_labels += len(lines)

            # Write the corrected lines back to the file
            with open(label_file, 'w') as file:
                file.write("\n".join(corrected_lines) + "\n")

        # Calculate error percentage
        error_percentage = (error_count / total_labels) * 100 if total_labels else 0
        error_df.loc[experiment_name, class_name] = error_percentage

    # Save the updated DataFrame to the CSV file
    error_df.to_csv(error_csv_path)


def split_data(split_ratio=0.8):
    TRAIN_IMAGES = os.path.join(TRAIN_PATH, "images")
    TRAIN_LABELS = os.path.join(TRAIN_PATH, "labels")
    VALID_IMAGES = os.path.join(VALID_PATH, "images")
    VALID_LABELS = os.path.join(VALID_PATH, "labels")
    os.makedirs(TRAIN_IMAGES, exist_ok=True)
    os.makedirs(TRAIN_LABELS, exist_ok=True)
    os.makedirs(VALID_IMAGES, exist_ok=True)
    os.makedirs(VALID_LABELS, exist_ok=True)

    for class_name in CLASSES:
        # Define paths
        class_image_dir = os.path.join(ACTIVE_LEARNING_PATH, class_name)
        class_label_dir = os.path.join(INFER_PATH, class_name)
        images = glob.glob(os.path.join(class_image_dir, "*.jpg"))
        labels = glob.glob(os.path.join(class_label_dir, "*.txt"))

        # Ensure corresponding image-label pairs
        base_names = [os.path.splitext(os.path.basename(image))[0] for image in images]
        images, labels = zip(
            *[(image, os.path.join(class_label_dir, base_name + ".txt")) for image, base_name in zip(images, base_names)
              if os.path.exists(os.path.join(class_label_dir, base_name + ".txt"))])

        # Split data
        images_train, images_val, labels_train, labels_val = train_test_split(images, labels, train_size=split_ratio,
                                                                              random_state=42)

        # Move files to respective directories
        for image, label in zip(images_train, labels_train):
            shutil.move(image, TRAIN_IMAGES)
            shutil.move(label, TRAIN_LABELS)
        for image, label in zip(images_val, labels_val):
            shutil.move(image, VALID_IMAGES)
            shutil.move(label, VALID_LABELS)

        # Print out counts
        print(f"Class {class_name}: Moved {len(images_train)} to train, {len(images_val)} to validation")

def count_images_in_active_learning_path():
    total_image_count = 0

    # Iterate over each class directory in ACTIVE_LEARNING_PATH
    for class_name in CLASSES:
        class_dir = os.path.join(ACTIVE_LEARNING_PATH, class_name)
        image_files = glob.glob(os.path.join(class_dir, "*.jpg"))  # Assuming images are in .jpg format
        total_image_count += len(image_files)

    return total_image_count