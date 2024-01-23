import os
import shutil
import random
from collections import defaultdict

# Assuming you have these paths set up already
from __dataset_path import TRAIN100_PATH, TRAIN_PATH, VALID_PATH

# Function to split data and move files
def split_data(class_files, train_ratio=0.8):
    # Shuffle files for each class
    for class_name, files in class_files.items():
        random.shuffle(files)

        # Split files based on the train_ratio
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        valid_files = files[split_idx:]

        # Move files to the respective TRAIN and VALID directories
        for file in train_files:
            shutil.move(os.path.join(TRAIN100_PATH, 'images', file), os.path.join(TRAIN_PATH, 'images', file))
            shutil.move(os.path.join(TRAIN100_PATH, 'labels', file.replace('.jpg', '.txt')),
                        os.path.join(TRAIN_PATH, 'labels', file.replace('.jpg', '.txt')))

        for file in valid_files:
            # Rename the file by replacing "TRAIN" with "VALID"
            valid_image_file = file.replace('TRAIN', 'VALID')
            valid_label_file = valid_image_file.replace('.jpg', '.txt')

            shutil.move(os.path.join(TRAIN100_PATH, 'images', file), os.path.join(VALID_PATH, 'images', valid_image_file))
            shutil.move(os.path.join(TRAIN100_PATH, 'labels', file.replace('.jpg', '.txt')),
                        os.path.join(VALID_PATH, 'labels', valid_label_file))


# Collect basenames of the files, grouped by class
class_files = defaultdict(list)
for filename in os.listdir(os.path.join(TRAIN100_PATH, 'images')):
    class_name = filename.split('_', 2)[0]  # Extract the class name
    class_files[class_name].append(filename)

# Ensure the destination directories exist
os.makedirs(os.path.join(TRAIN_PATH, 'images'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_PATH, 'labels'), exist_ok=True)
os.makedirs(os.path.join(VALID_PATH, 'images'), exist_ok=True)
os.makedirs(os.path.join(VALID_PATH, 'labels'), exist_ok=True)

# Split and move the files
split_data(class_files)