import os
import yaml
from __dataset_path import *

# Load class names from the YAML file and create a mapping from class name to index
with open(YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    class_name_to_index = {name: index for index, name in enumerate(data['names'])}

# Directory containing the re-labeled files
RE_LABEL_DIR = os.path.join(DATASET_DIR, 're_label')

# Function to update the class number in label files and check for emptiness
def update_label_files():
    labels_path = os.path.join(RE_LABEL_DIR, 'labels')
    empty_label_files = []  # List to keep track of empty label files
    for label_file in os.listdir(labels_path):
        file_path = os.path.join(labels_path, label_file)
        # Extract the class name from the filename (up to the first underscore)
        class_name = label_file.split('_', 1)[0]
        # Get the class index
        class_index = class_name_to_index.get(class_name)
        if class_index is not None:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            if not lines:
                empty_label_files.append(file_path)
            with open(file_path, 'w') as file:
                for line in lines:
                    parts = line.split(' ')
                    # Change the class number to the correct index
                    parts[0] = str(class_index)
                    # Write the updated line back to the file
                    file.write(' '.join(parts))
    return empty_label_files

# Update label files in the re-label folder and check for empty files
empty_label_files = update_label_files()

# Report the results
if empty_label_files:
    print("Empty label files detected after correction:")
    for file_path in empty_label_files:
        print(file_path)
else:
    print("No empty label files detected after correction.")
