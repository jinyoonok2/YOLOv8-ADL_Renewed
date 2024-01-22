import os
from dataset_path import RELABEL_PATH, TRAIN100_PATH


# Function to rename files in a directory
def rename_files(directory, extension, label_directory=None):
    encountered_names = set()

    for filename in os.listdir(directory):
        if filename.endswith(extension):
            # Split the filename on '_jpg'
            parts = filename.split('_jpg')

            # If '_jpg' is not found or it's already the proper format, skip renaming
            if len(parts) == 1 or (len(parts) == 2 and parts[1] == extension):
                print(f"Skipped '{filename}' (no change needed)")
                continue

            new_name = parts[0] + extension
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_name)

            if new_name not in encountered_names:
                # Rename the file if the new name hasn't been encountered before
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_name}'")
                encountered_names.add(new_name)
            else:
                # Remove the file (and its corresponding label file if it's an image) if the new name already exists
                os.remove(old_file_path)
                print(f"Removed duplicate file '{old_file_path}'")

                # If it's an image file and a label directory is provided, remove the corresponding label file
                if extension == '.jpg' and label_directory is not None:
                    label_filename = new_name.replace('.jpg', '.txt')
                    label_file_path = os.path.join(label_directory, label_filename)
                    if os.path.exists(label_file_path):
                        os.remove(label_file_path)
                        print(f"Removed corresponding label file '{label_file_path}'")


# Directories containing the re-labeled files
# re_label_images_path = os.path.join(RELABEL_PATH, 'images')
# re_label_labels_path = os.path.join(RELABEL_PATH, 'labels')
re_label_images_path = os.path.join(TRAIN100_PATH, 'images')
re_label_labels_path = os.path.join(TRAIN100_PATH, 'labels')

# Rename files in the images and labels directories
rename_files(re_label_images_path, '.jpg')
rename_files(re_label_labels_path, '.txt')

print("Renaming completed.")
