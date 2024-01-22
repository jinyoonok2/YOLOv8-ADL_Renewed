import os
import re
import yaml
from collections import defaultdict
from dataset_path import *

# Load class names from a YAML file
with open(YAML_PATH, 'r') as file:
    class_names = yaml.safe_load(file)

old_new_names_map = dict(zip(class_names['old_names'], class_names['names']))

# Separate counters for training and testing datasets
file_counters = {
    'TRAIN': defaultdict(int),
    'TEST': defaultdict(int)
}

# Counter to keep track of the file numbering for each class
active_file_counters = defaultdict(int)
# Dictionary to store the count of images for each class before and after renaming
image_counts_before_after = defaultdict(lambda: {'before': 0, 'after': 0})

# TOMATO
def rename_files_in_directory(dataset_type, image_dir, label_dir, image_extension, label_extension):
    for filename in os.listdir(image_dir):
        # Check if any old class name is in the filename
        for old_name, new_name in old_new_names_map.items():
            if old_name in filename:
                # Increment the counter for this class in the appropriate dataset and generate the new file name
                file_counters[dataset_type][new_name] += 1
                new_image_filename = f"{new_name}_{dataset_type}_{file_counters[dataset_type][new_name]}{image_extension}"
                new_label_filename = f"{new_name}_{dataset_type}_{file_counters[dataset_type][new_name]}{label_extension}"

                # Rename the image file
                old_image_path = os.path.join(image_dir, filename)
                new_image_path = os.path.join(image_dir, new_image_filename)
                os.rename(old_image_path, new_image_path)

                # Rename the corresponding label file
                label_filename = filename.replace(image_extension, label_extension)
                old_label_path = os.path.join(label_dir, label_filename)
                new_label_path = os.path.join(label_dir, new_label_filename)
                if os.path.exists(old_label_path):
                    os.rename(old_label_path, new_label_path)

                print(f"Renamed image '{old_image_path}' to '{new_image_path}'")
                print(f"Renamed label '{old_label_path}' to '{new_label_path}'")
                break  # Break the loop after finding and replacing the first occurrence



# APPLE, replace '___' with '_'
def replace_underscores_in_filenames(image_dir, label_dir, image_extension, label_extension):
    for dir_path in [image_dir, label_dir]:
        for filename in os.listdir(dir_path):
            if '___' in filename:
                # Replace '___' with '_' in the filename
                new_filename = filename.replace('___', '_')

                # Rename the file
                old_file_path = os.path.join(dir_path, filename)
                new_file_path = os.path.join(dir_path, new_filename)
                os.rename(old_file_path, new_file_path)

                print(f"Renamed file '{old_file_path}' to '{new_file_path}'")


def count_files(directory, file_extension):
    count_dict = defaultdict(int)
    for filename in os.listdir(directory):
        for new_name in old_new_names_map.values():
            if filename.startswith(new_name) and filename.endswith(file_extension):
                count_dict[new_name] += 1
    return count_dict

def rename_active_learning_folders(active_learning_dir):
    # Step 1: Rename folders according to new class names
    for folder in os.listdir(active_learning_dir):
        if folder in old_new_names_map:
            new_folder_name = old_new_names_map[folder]
            old_folder_path = os.path.join(active_learning_dir, folder)
            new_folder_path = os.path.join(active_learning_dir, new_folder_name)
            if not os.path.exists(new_folder_path):  # Check if new folder already exists
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed folder '{old_folder_path}' to '{new_folder_path}'")

def rename_active_learning_files(active_learning_dir, image_extension):
    # Step 2: Rename image files in each folder
    for new_folder_name in old_new_names_map.values():
        folder_path = os.path.join(active_learning_dir, new_folder_name)
        if os.path.exists(folder_path):
            # Count images before renaming
            image_counts_before_after[new_folder_name]['before'] = len(
                [f for f in os.listdir(folder_path) if f.endswith(image_extension)])

            for filename in os.listdir(folder_path):
                if filename.endswith(image_extension):
                    active_file_counters[new_folder_name] += 1
                    new_filename = f"{new_folder_name}_{active_file_counters[new_folder_name]}{image_extension}"
                    old_file_path = os.path.join(folder_path, filename)
                    new_file_path = os.path.join(folder_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed image '{old_file_path}' to '{new_file_path}'")

            # Count images after renaming
            image_counts_before_after[new_folder_name]['after'] = active_file_counters[new_folder_name]


# Print image counts before and after renaming
def print_image_counts():
    for class_name, counts in image_counts_before_after.items():
        print(f"Class {class_name}: {counts['before']} images before, {counts['after']} images after renaming")

def count_files_by_class(image_dir, old_new_names_map, image_extension):
    # Initialize a dictionary to hold the count for each class
    class_counts = {old_name: 0 for old_name in old_new_names_map.keys()}

    # Iterate over all files in the directory
    for filename in os.listdir(image_dir):
        # Check if the filename contains any of the old class names
        for old_name in old_new_names_map.keys():
            if old_name in filename and filename.endswith(image_extension):
                class_counts[old_name] += 1
                break  # Stop checking other class names if the current one matches

    return class_counts

# For APPLE
# replace_underscores_in_filenames(os.path.join(TRAIN100_PATH, 'images'), os.path.join(TRAIN100_PATH, 'labels'), '.jpg', '.txt')
# replace_underscores_in_filenames(os.path.join(TEST_PATH, 'images'), os.path.join(TEST_PATH, 'labels'), '.jpg', '.txt')

# For APPLE: Use the function to count files for each class in your image directories
# train_class_counts = count_files_by_class(os.path.join(TRAIN100_PATH, 'images'), old_new_names_map, '.jpg')
# test_class_counts = count_files_by_class(os.path.join(TEST_PATH, 'images'), old_new_names_map, '.jpg')
#
# print("Counts for each class in TRAIN images:", train_class_counts)
# print("Counts for each class in TEST images:", test_class_counts)


# Rename files in TRAIN and TEST datasets
rename_files_in_directory('TRAIN', os.path.join(TRAIN100_PATH, 'images'), os.path.join(TRAIN100_PATH, 'labels'), '.jpg',
                          '.txt')
rename_files_in_directory('TEST', os.path.join(TEST_PATH, 'images'), os.path.join(TEST_PATH, 'labels'), '.jpg', '.txt')

# Step 1: Rename folders
rename_active_learning_folders(ACTIVE_LEARNING_PATH)

# Step 2: Rename files in renamed folders
rename_active_learning_files(ACTIVE_LEARNING_PATH, '.jpg')

# Count files after renaming
train_image_counts = count_files(os.path.join(TRAIN100_PATH, 'images'), '.jpg')
test_image_counts = count_files(os.path.join(TEST_PATH, 'images'), '.jpg')
train_label_counts = count_files(os.path.join(TRAIN100_PATH, 'labels'), '.txt')
test_label_counts = count_files(os.path.join(TEST_PATH, 'labels'), '.txt')

# Print the counts for each class
print("Counts for each class in TRAIN images:", train_image_counts)
print("Counts for each class in TEST images:", test_image_counts)
print("Counts for each class in TRAIN labels:", train_label_counts)
print("Counts for each class in TEST labels:", test_label_counts)

# Print the image counts before and after renaming
print_image_counts()