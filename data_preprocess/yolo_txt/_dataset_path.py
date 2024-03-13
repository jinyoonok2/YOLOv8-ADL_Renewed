import os
import yaml

# TOMATO
# DATASET_DIR = r"C:\Jinyoon Projects\datasets\PlantVillage_tomato"
# YAML_PATH = os.path.join(DATASET_DIR, "TOMATO_DATA_YAML.yaml")
# TRAIN100_PATH = os.path.join(DATASET_DIR, "TOMATO-TRAIN100-YOLO")
#
# TRAIN_PATH = os.path.join(DATASET_DIR, "TOMATO-TRAIN-YOLO")
# VALID_PATH = os.path.join(DATASET_DIR, "TOMATO-VALID-YOLO")
# TEST_PATH = os.path.join(DATASET_DIR, "TOMATO-TEST-YOLO")
#
# ACTIVE_LEARNING_PATH = os.path.join(DATASET_DIR, "TOMATO_ACTIVE_LEARNING")
# RELABEL_PATH = os.path.join(DATASET_DIR, "re_label")
#
# PROJECT_PATH = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\tomato_segment'
# INFER_PATH = os.path.join(DATASET_DIR, "TOMATO_PRED_RESULT")

# APPLE
# DATASET_DIR = r"C:\Jinyoon Projects\datasets\PlantVillage_apple"
# YAML_PATH = os.path.join(DATASET_DIR, "APPLE_DATA_YAML.yaml")
# TRAIN100_PATH = os.path.join(DATASET_DIR, "APPLE-TRAIN20-YOLO")
#
# TRAIN_PATH = os.path.join(DATASET_DIR, "APPLE-TRAIN-YOLO")
# VALID_PATH = os.path.join(DATASET_DIR, "APPLE-VALID-YOLO")
# TEST_PATH = os.path.join(DATASET_DIR, "APPLE-TEST-YOLO")
#
# ACTIVE_LEARNING_PATH = os.path.join(DATASET_DIR, "APPLE_ACTIVE_LEARNING")
# RELABEL_PATH = os.path.join(DATASET_DIR, "re_label")
#
# PROJECT_PATH = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\apple_segment'
# INFER_PATH = os.path.join(DATASET_DIR, "APPLE_PRED_RESULT")


# HAM
HAM_10000_PATH_IMAGE = r"C:\Jinyoon Projects\datasets\HAM10000\train\images"
HAM_10000_PATH_LABEL = r"C:\Jinyoon Projects\datasets\HAM10000\train\labels"

DATASET_DIR = r"C:\Jinyoon Projects\datasets\HAM10000"
YAML_PATH = os.path.join(DATASET_DIR, "HAM_DATA_YAML.yaml")
TRAIN100_PATH = os.path.join(DATASET_DIR, "HAM-TRAIN60-YOLO")

TRAIN_PATH = os.path.join(DATASET_DIR, "HAM-TRAIN-YOLO")
VALID_PATH = os.path.join(DATASET_DIR, "HAM-VALID-YOLO")
TEST_PATH = os.path.join(DATASET_DIR, "HAM-TEST-YOLO")

ACTIVE_LEARNING_PATH = os.path.join(DATASET_DIR, "HAM_ACTIVE_LEARNING")
RELABEL_PATH = os.path.join(DATASET_DIR, "re_label")

PROJECT_PATH = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\ham_segment'
INFER_PATH = os.path.join(DATASET_DIR, "HAM_PRED_RESULT")

# COMMON
# Load class names from a YAML file
with open(YAML_PATH, 'r') as file:
    class_names = yaml.safe_load(file)
# Assuming the class names are under the 'names' key in your YAML file
CLASSES = class_names['names']
# Create id2label dictionary
ID2LABEL = {idx: label for idx, label in enumerate(CLASSES)}

TEST_DATA_DIR = r"C:\Jinyoon Projects\datasets"
TEST_HAM_PATH = os.path.join(TEST_DATA_DIR, r"HAM10000\HAM_DATA_YAML.yaml")
TEST_APPLE_PATH = os.path.join(TEST_DATA_DIR, r"PlantVillage_apple\APPLE_DATA_YAML.yaml")
TEST_TOMATO_PATH = os.path.join(TEST_DATA_DIR, r"PlantVillage_tomato\TOMATO_DATA_YAML.yaml")

TEST_APPLE_MODEL = os.path.join(r"C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\apple_segment")
TEST_TOMATO_MODEL = os.path.join(r"C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\tomato_segment")
TEST_HAM_MODEL = os.path.join(r"C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\ham_segment")
