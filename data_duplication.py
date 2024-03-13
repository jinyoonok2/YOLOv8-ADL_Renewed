from PIL import Image
import os


def duplicate_files_in_subdirs(input_dir, num_copies=2):
    """
    Duplicate each file in the 'images' and 'masks' subdirectories of the given directory
    'num_copies' times with a specific naming convention.

    Args:
    - input_dir (str): The directory containing the 'images' and 'masks' subdirectories.
    - num_copies (int): The number of duplicates to create for each file.
    """
    # Define the subdirectories
    images_dir = os.path.join(input_dir, 'images')
    masks_dir = os.path.join(input_dir, 'masks')

    # Process each subdirectory
    for subdir in [images_dir, masks_dir]:
        for file in os.listdir(subdir):
            file_path = os.path.join(subdir, file)
            if subdir == images_dir and file.endswith('.jpg'):
                base_name = file[:-4]  # Remove '.jpg' extension
                extension = '.jpg'
            elif subdir == masks_dir and file.endswith('_mask.png'):
                base_name = file[:-9]  # Remove '_mask.png' extension
                extension = '_mask.png'
            else:
                continue  # Skip files that do not match the expected extensions

            # Load the original file
            original = Image.open(file_path)

            # Create duplicates with the specific naming convention
            for i in range(1, num_copies + 1):
                if extension == '.jpg':
                    new_file_name = f"{base_name}_{i}{extension}"
                else:  # For mask files
                    new_file_name = f"{base_name}-{i}{extension}"
                new_file_path = os.path.join(subdir, new_file_name)
                original.save(new_file_path)
                print(f"Saved duplicate: {new_file_path}")


def duplicate_images_and_labels(input_dir, num_copies=4):
    """
    Duplicate each file in the 'images' subdirectory and their corresponding text label
    files in the 'labels' subdirectory 'num_copies' times with a specific naming convention.

    Args:
    - input_dir (str): The directory containing the 'images' and 'labels' subdirectories.
    - num_copies (int): The number of duplicates to create for each file.
    """
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')

    for subdir in [images_dir, labels_dir]:
        for file in os.listdir(subdir):
            file_path = os.path.join(subdir, file)
            if subdir == images_dir and file.endswith('.jpg'):
                base_name = file[:-4]  # Remove '.jpg' extension
                extension = '.jpg'
            elif subdir == labels_dir and file.endswith('.txt'):
                base_name = file[:-4]  # Remove '.txt' extension
                extension = '.txt'
            else:
                continue  # Skip files that do not match the expected extensions

            if extension == '.jpg':
                # Load the original image
                original = Image.open(file_path)
            else:
                # Open the original text file
                with open(file_path, 'r') as original:
                    content = original.read()

            # Create duplicates with the specific naming convention
            for i in range(1, num_copies + 1):
                new_file_name = f"{base_name}_{i}{extension}"
                new_file_path = os.path.join(subdir, new_file_name)

                if extension == '.jpg':
                    original.save(new_file_path)
                else:  # For text label files
                    with open(new_file_path, 'w') as new_file:
                        new_file.write(content)
                print(f"Saved duplicate: {new_file_path}")


# Example usage
input_dir = r"C:\Jinyoon Projects\datasets\HAM10000\HAM-TRAIN-YOLO"  # Update with the actual path to your directory
# duplicate_files_in_subdirs(input_dir)
duplicate_images_and_labels(input_dir)
