import os
from glob import glob
import glob
import shutil
import yaml
import random

def correct_data(LABEL_PATH, label_map):
    # Check if the path exists and is a directory
    if not os.path.exists(LABEL_PATH) or not os.path.isdir(LABEL_PATH):
        print(f"No such directory: {LABEL_PATH}")
        return

    reverse_label_map = {v: k for k, v in label_map.items()}  # Reverse mapping

    # 1. detect the error, then correct them into correct label class
    for disease in label_map.values():
        dir_path = os.path.join(LABEL_PATH, disease)

        # Get all text files in the directory
        text_files = glob.glob(os.path.join(dir_path, f"{disease}_*.txt"))
        num_errors = 0

        # If there are no labels in the directory, print a notification and skip to the next disease
        if not text_files:
            print(f"No labels found in directory: {dir_path}")
            continue

        total_labels = len(text_files)

        # Iterate over each text file
        for file_path in text_files:
            with open(file_path, "r") as f:
                contents = f.read()
                first_char = contents[0]

                # Check if the first character matches the corresponding value in reverse_label_map
                if int(first_char) != reverse_label_map[disease]:
                    num_errors += 1

                    # Replace the first character with the corresponding value from the dictionary
                    contents = str(reverse_label_map[disease]) + contents[1:]
                    with open(file_path, "w") as f2:
                        f2.write(contents)

        error_percentage = ((total_labels - num_errors) / total_labels) * 100
        print(f"{disease}: {num_errors} errors out of {total_labels}: ({error_percentage:.2f}% of labels success)")


def split_data(LABEL_PATH, IMG_PATH, label_map, split_ratio=0.8):

    diseases = label_map.values()
    UNANNOTATED_IMAGES = os.path.join(LABEL_PATH, "unannotated")

    # Create subdirectories for images and labels
    TRAIN_IMAGES = os.path.join(LABEL_PATH, "train/images")
    TRAIN_LABELS = os.path.join(LABEL_PATH, "train/labels")
    VALID_IMAGES = os.path.join(LABEL_PATH, "valid/images")
    VALID_LABELS = os.path.join(LABEL_PATH, "valid/labels")
    os.makedirs(TRAIN_IMAGES, exist_ok=True)
    os.makedirs(TRAIN_LABELS, exist_ok=True)
    os.makedirs(VALID_IMAGES, exist_ok=True)
    os.makedirs(VALID_LABELS, exist_ok=True)

    for disease in diseases:
        img_dir_path = os.path.join(IMG_PATH, disease)
        labels_dir_path = os.path.join(LABEL_PATH, disease)

        # Create disease directories under unannotated
        disease_unannotated = os.path.join(UNANNOTATED_IMAGES, disease)
        os.makedirs(disease_unannotated, exist_ok=True)

        # Check if the label directory is empty, if so skip the current disease
        if not os.listdir(labels_dir_path):
            print(f"no detected labels found in {labels_dir_path}")
            image_files = glob.glob(os.path.join(img_dir_path, "*.jpg"))
            for image_file in image_files:
                dst_file_path = os.path.join(disease_unannotated, os.path.basename(image_file))
                if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                    shutil.copy(image_file, dst_file_path)
            continue

        # Get all image files and their corresponding label files in the directory
        image_files = glob.glob(os.path.join(img_dir_path, "*.jpg"))
        label_files = glob.glob(os.path.join(labels_dir_path, "*.txt"))

        # Collect all base names (without extension) for label files
        label_names = [os.path.splitext(os.path.basename(label_file))[0] for label_file in label_files]

        num_labels_to_move = int(split_ratio * len(label_files))

        # Split the label files into training and validation sets
        train_labels = label_files[:num_labels_to_move]
        valid_labels = label_files[num_labels_to_move:]

        # Copy label files to corresponding directories
        for label_file in train_labels:
            dst_file_path = os.path.join(TRAIN_LABELS, os.path.basename(label_file))
            if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                shutil.copy(label_file, dst_file_path)

        for label_file in valid_labels:
            dst_file_path = os.path.join(VALID_LABELS, os.path.basename(label_file))
            if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                shutil.copy(label_file, dst_file_path)

        # Check each image file
        for image_file in image_files:
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            # Check if there is a corresponding label file
            if image_name in label_names:
                if image_name + '.txt' in [os.path.basename(label) for label in train_labels]:
                    dst_file_path = os.path.join(TRAIN_IMAGES, os.path.basename(image_file))
                    if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                        shutil.copy(image_file, dst_file_path)

                elif image_name + '.txt' in [os.path.basename(label) for label in valid_labels]:
                    dst_file_path = os.path.join(VALID_IMAGES, os.path.basename(image_file))
                    if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                        shutil.copy(image_file, dst_file_path)
            else:
                # If no corresponding label exists, move the image to the unannotated disease directory
                dst_file_path = os.path.join(disease_unannotated, os.path.basename(image_file))
                if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                    shutil.copy(image_file, dst_file_path)

def recall_existing_data(prev_data_path, label_path, label_map, split_ratio = 0.8):
    sub_dirs = ["train", "valid"]
    dir_names = ["labels", "images"]
    label_map = list(label_map.values())

    all_files = []
    for sub_dir in sub_dirs:
        orig_label_path = os.path.join(prev_data_path, sub_dir, dir_names[0])
        orig_image_path = os.path.join(prev_data_path, sub_dir, dir_names[1])
        label_files = glob.glob(os.path.join(orig_label_path, "*.txt"))
        image_files = [os.path.join(orig_image_path, os.path.splitext(os.path.basename(lf))[0] + ".jpg") for lf in
                       label_files]
        all_files.extend(list(zip(label_files, image_files)))

    random.shuffle(all_files)

    num_train = int(len(all_files) * split_ratio)
    train_files = all_files[:num_train]
    valid_files = all_files[num_train:]

    def copy_files(files, sub_dir):
        new_label_path = os.path.join(label_path, sub_dir, dir_names[0])
        new_image_path = os.path.join(label_path, sub_dir, dir_names[1])
        os.makedirs(new_label_path, exist_ok=True)
        os.makedirs(new_image_path, exist_ok=True)
        for lf, imf in files:
            base_filename = os.path.splitext(os.path.basename(lf))[0]
            if "_jpg.rf." in base_filename:
                cls_name = '_'.join(base_filename.split('_')[:-2])
            else:
                cls_name = '_'.join(base_filename.split('_')[:-1])
            if cls_name not in label_map:
                print(f"Warning: Class name '{cls_name}' extracted from label file '{lf}' not found in label map. Skipping this file.")
                continue
            new_label_file = os.path.join(new_label_path, base_filename + ".txt")
            new_image_file = os.path.join(new_image_path, base_filename + ".jpg")
            shutil.copy(lf, new_label_file)
            shutil.copy(imf, new_image_file)

    copy_files(train_files, sub_dirs[0])
    copy_files(valid_files, sub_dirs[1])

def generate_yaml(label_path, label_map):
    sorted_label_map = dict(sorted(label_map.items()))  # Sort label_map by keys
    class_names = list(sorted_label_map.values())  # Extract class names in the order of keys

    # Create the dictionary to save as a YAML file
    data = {
        'names': class_names,
        'nc': len(class_names),  # Number of classes
        'train': './train',
        'val': './valid',
    }

    # Generate the YAML file path
    yaml_file_path = os.path.join(label_path, 'data.yaml')

    # Save the dictionary as a YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

    print(f"YAML file has been saved to {yaml_file_path}")
