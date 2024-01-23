import os
from __dataset_path import TRAIN100_PATH, TEST_PATH, YAML_PATH
import yaml

# Load class names from a YAML file
with open(YAML_PATH, 'r') as file:
    class_names = yaml.safe_load(file)

# Use 'names' for counting
new_names = class_names['names']

def count_files_by_new_names(directory, new_names, file_extension):
    # Initialize a dictionary to hold the count for each class
    class_counts = {new_name: 0 for new_name in new_names}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename contains any of the new class names
        for new_name in new_names:
            if new_name in filename and filename.endswith(file_extension):
                class_counts[new_name] += 1
                break  # Stop checking other class names if the current one matches

    return class_counts

# Use the function to count files for each class in your image and label directories
train_image_class_counts = count_files_by_new_names(os.path.join(TRAIN100_PATH, 'images'), new_names, '.jpg')
test_image_class_counts = count_files_by_new_names(os.path.join(TEST_PATH, 'images'), new_names, '.jpg')

train_label_class_counts = count_files_by_new_names(os.path.join(TRAIN100_PATH, 'labels'), new_names, '.txt')
test_label_class_counts = count_files_by_new_names(os.path.join(TEST_PATH, 'labels'), new_names, '.txt')

print("Counts for each class in TRAIN images:", train_image_class_counts)
print("Counts for each class in TEST images:", test_image_class_counts)
print("Counts for each class in TRAIN labels:", train_label_class_counts)
print("Counts for each class in TEST labels:", test_label_class_counts)
