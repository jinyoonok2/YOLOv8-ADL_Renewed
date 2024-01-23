import yaml
import os
import shutil
from __dataset_path import TRAIN100_PATH, TEST_PATH, ACTIVE_LEARNING_PATH, YAML_PATH


def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']


def distribute_files():
    # Load class names
    class_names = load_class_names(YAML_PATH)

    # Initialize dictionary to hold base names for each class
    class_files = {class_name: {'images': [], 'labels': []} for class_name in class_names}

    # Collect image and label filenames for each class
    for filename in os.listdir(os.path.join(TRAIN100_PATH, 'images')):
        basename, _ = os.path.splitext(filename)
        for class_name in class_names:
            if class_name in filename:
                class_files[class_name]['images'].append(basename)
                corresponding_label = basename + '.txt'
                if corresponding_label in os.listdir(os.path.join(TRAIN100_PATH, 'labels')):
                    class_files[class_name]['labels'].append(basename)
                break

    # Move 10% of files to TEST_PATH
    for class_name, files in class_files.items():
        num_files = len(files['images'])
        num_test_files = max(1, num_files // 10)  # Ensure at least one file

        for i in range(num_test_files):
            image_name = files['images'][i] + '.jpg'
            label_name = files['labels'][i] + '.txt'
            shutil.move(os.path.join(TRAIN100_PATH, 'images', image_name),
                        os.path.join(TEST_PATH, 'images', image_name.replace('.jpg', '_TEST.jpg')))
            shutil.move(os.path.join(TRAIN100_PATH, 'labels', label_name),
                        os.path.join(TEST_PATH, 'labels', label_name.replace('.txt', '_TEST.txt')))
            print(f"Moved {image_name} and {label_name} to TEST_PATH")

        # Create class-named folders under ACTIVE_LEARNING_PATH
        active_learning_class_path = os.path.join(ACTIVE_LEARNING_PATH, class_name)
        if not os.path.exists(active_learning_class_path):
            os.makedirs(active_learning_class_path)

        # Move remaining images (except last 60) to ACTIVE_LEARNING_PATH
        for i in range(num_test_files, num_files - 60):
            image_name = files['images'][i] + '.jpg'
            label_name = files['labels'][i] + '.txt'
            shutil.move(os.path.join(TRAIN100_PATH, 'images', image_name),
                        os.path.join(active_learning_class_path, image_name))
            os.remove(os.path.join(TRAIN100_PATH, 'labels', label_name))
            print(f"Moved {image_name} to ACTIVE_LEARNING_PATH and removed {label_name} from TRAIN100_PATH/labels")

        # Leave last 60 files in TRAIN100_PATH (no action needed here)
        print(f"Left last 60 images and labels of {class_name} in TRAIN100_PATH")


distribute_files()