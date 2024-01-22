from dataset_path import HAM_10000_PATH_IMAGE, HAM_10000_PATH_LABEL, YAML_PATH
import os
from PIL import Image
import yaml


def remove_dark_images(image_dir, label_dir, class_names):
    # Load class names from a YAML file
    with open(class_names, 'r') as file:
        data = yaml.safe_load(file)

    # Names from YAML except 'NV'
    valid_class_names = [name for name in data['names'] if name != 'NV']

    # Iterate through files in the image directory
    for filename in os.listdir(image_dir):
        # Check if the file is an image and has a valid class name
        if any(name in filename for name in valid_class_names) and filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            pixels = list(image.getdata())

            # Check for pure dark pixels
            if any(pixel == (0, 0, 0) for pixel in pixels):
                # Remove the image file
                os.remove(image_path)
                print(f"Removed image: {image_path}")

                # Remove the corresponding label file
                basename = os.path.splitext(filename)[0]
                label_path = os.path.join(label_dir, basename + '.txt')
                if os.path.exists(label_path):
                    os.remove(label_path)
                    print(f"Removed label: {label_path}")

remove_dark_images(HAM_10000_PATH_IMAGE, HAM_10000_PATH_LABEL, YAML_PATH)
