import os
import yaml

DATASET_DIR = r"C:\Jinyoon Projects\datasets\PlantVillage_apple_mask"

YAML_PATH = os.path.join(DATASET_DIR, "APPLE_DATA_YAML.yaml")

APPLE_TRAIN_PATH = os.path.join(DATASET_DIR, "APPLE-TRAIN-YOLO")
TRAIN_PATH_IMAGE = os.path.join(APPLE_TRAIN_PATH, "images")
TRAIN_PATH_MASK = os.path.join(APPLE_TRAIN_PATH, "masks")
TRAIN_PATH_LABEL = os.path.join(APPLE_TRAIN_PATH, "labels")

APPLE_TEST_PATH = os.path.join(DATASET_DIR, "APPLE-TEST-YOLO")
TEST_PATH_IMAGE = os.path.join(APPLE_TEST_PATH, "images")
TEST_PATH_MASK = os.path.join(APPLE_TEST_PATH, "masks")
TEST_PATH_LABEL = os.path.join(APPLE_TEST_PATH, "labels")

ACTIVE_LEARNING_PATH = os.path.join(DATASET_DIR, "APPLE_ACTIVE_LEARNING")

# Load class names from a YAML file
with open(YAML_PATH, 'r') as file:
    class_names = yaml.safe_load(file)
# Assuming the class names are under the 'names' key in your YAML file
CLASSES = class_names['names']
# Create id2label dictionary
ID2LABEL = {idx: label for idx, label in enumerate(CLASSES)}