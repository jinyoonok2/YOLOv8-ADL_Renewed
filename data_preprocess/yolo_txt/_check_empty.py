import os
import shutil
from data_preprocess.yolo_txt.dataset_path import DATASET_DIR, TRAIN_PATH, VALID_PATH, TEST_PATH

############################################################################################
# Check Version - train/valid/test
############################################################################################
# # Directories to check
# directories_to_check = [TRAIN_PATH, VALID_PATH, TEST_PATH]
#
# # Function to check for empty files in a directory
# def check_empty_files(directory):
#     labels_path = os.path.join(directory, 'labels')
#     if os.path.exists(labels_path):
#         for filename in os.listdir(labels_path):
#             file_path = os.path.join(labels_path, filename)
#             # Check if the file is empty
#             if os.path.getsize(file_path) == 0:
#                 print(f"Empty label file: {file_path}")
#
# # Iterate over train, valid, and test directories and check for empty label files
# for directory in directories_to_check:
#     os.path.join(DATASET_DIR, directory)
#     check_empty_files(directory)
#
# print("Check completed.")
############################################################################################
# Move Version - train/valid/test
############################################################################################
# Directory to move empty labels and corresponding images
RE_LABEL_DIR = os.path.join(DATASET_DIR, 're_label')


# Function to check for empty files in a directory and move them
def check_empty_files_and_move(directory):
    labels_path = os.path.join(directory, 'labels')
    images_path = os.path.join(directory, 'images')

    # Ensure the re-label directory and its subdirectories exist
    re_label_labels_path = os.path.join(RE_LABEL_DIR, 'labels')
    re_label_images_path = os.path.join(RE_LABEL_DIR, 'images')
    os.makedirs(re_label_labels_path, exist_ok=True)
    os.makedirs(re_label_images_path, exist_ok=True)

    if os.path.exists(labels_path):
        for filename in os.listdir(labels_path):
            label_file_path = os.path.join(labels_path, filename)
            image_file_path = os.path.join(images_path, filename.replace('.txt', '.jpg'))

            # Check if the label file is empty
            if os.path.getsize(label_file_path) == 0:
                print(f"Empty label file: {label_file_path}")
                print(f"Corresponding image: {image_file_path}")

                # Move the empty label file and corresponding image to the re-label directory
                shutil.move(label_file_path, os.path.join(re_label_labels_path, filename))
                if os.path.exists(image_file_path):
                    shutil.move(image_file_path, os.path.join(re_label_images_path, filename.replace('.txt', '.jpg')))


# Iterate over train, valid, and test directories and check for empty label files
for directory in [TRAIN_PATH, VALID_PATH, TEST_PATH]:
    check_empty_files_and_move(directory)

print("Check and move completed.")