import os

DATASET_DIR = r"C:\Jinyoon Projects\datasets\PlantVillage_mask"

YAML_PATH = os.path.join(DATASET_DIR, "TOMATO_DATA_YAML.yaml")

TRAIN_PATH_MASK = os.path.join(DATASET_DIR, "TOMATO-TRAIN100-YOLO")
TEST_PATH_MASK = os.path.join(DATASET_DIR, "TOMATO-TEST-YOLO")

ACTIVE_LEARNING_PATH = os.path.join(DATASET_DIR, "TOMATO_ACTIVE_LEARNING")