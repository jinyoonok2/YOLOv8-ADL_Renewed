from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from transformers import AutoModelForUniversalSegmentation
from _oneformer_custom_data import processor
import numpy as np
import os
from tqdm import tqdm

def load_model(checkpoint_path):
    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)
    checkpoint = torch.load(checkpoint_path)
    # Directly load the state dictionary without accessing 'model_state_dict'
    model.load_state_dict(checkpoint)
    model.model.is_training = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


def evaluate_model(test_path, output_path, model, processor, device):
    images_dir = os.path.join(test_path, 'images')
    masks_dir = os.path.join(test_path, 'masks')

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    for image_file in tqdm(image_files):
        base_name = image_file.split('.')[0]  # Remove the file extension to get the base name

        # Define paths for the original image, its ground truth mask, and the predicted mask
        image_path = os.path.join(images_dir, image_file)
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(masks_dir, mask_name)

        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        semantic_segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[(256, 256)])[0]

        # Convert segmentation to binary mask using PyTorch operations
        output_mask = (semantic_segmentation > 0).to(torch.uint8)  # Convert all non-background pixels to 1

        # Move the mask to CPU and convert to NumPy for saving
        output_mask_np = output_mask.cpu().numpy() * 255

        # Load and convert the ground truth mask to a binary mask
        gt_mask = np.array(Image.open(mask_path))
        gt_binary_mask = (gt_mask > 0).astype(np.uint8) * 255  # Convert all non-background pixels to 1

        # Save the original image, ground truth binary mask, and output binary mask using a comprehensible naming convention
        image.save(os.path.join(output_path, f"{base_name}.jpg"))
        Image.fromarray(gt_binary_mask).save(os.path.join(output_path, f"{base_name}_test.png"))
        Image.fromarray(output_mask_np).save(os.path.join(output_path, f"{base_name}_result.png"))

        print(f"Processed {image_file}")


def convert_masks_to_binary(input_dir, output_dir):
    """
    Converts all mask PNG files in input_dir to binary masks and saves them to output_dir.

    Args:
    - input_dir (str): The directory containing the original mask PNG files.
    - output_dir (str): The directory where the binary mask PNG files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all PNG mask files in the input directory
    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    for mask_file in mask_files:
        # Construct the full path to the mask file
        mask_path = os.path.join(input_dir, mask_file)

        # Load the mask image
        mask = np.array(Image.open(mask_path))

        # Convert to binary mask: pixels > 0 to 1, else 0
        binary_mask = (mask > 0).astype(np.uint8)

        # Save the binary mask to the output directory
        binary_mask_image = Image.fromarray(binary_mask * 255)  # Convert to [0, 255] for saving
        binary_mask_image.save(os.path.join(output_dir, mask_file))

        print(f"Converted and saved binary mask for {mask_file}")


# Example usage
# input_dir = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\APPLE-TRAIN-YOLO\masks'  # Update with your actual path
# output_dir = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\output_result'  # Update with your actual path
#
# convert_masks_to_binary(input_dir, output_dir)

# # Example usage
checkpoint_path = r"C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\ADS_oneformer\one_former\train-final-6\final_model.pt"  # Update with your actual path
test_path = r'C:\Jinyoon Projects\datasets\PlantVillage_apple_mask\APPLE-TEST-YOLO'  # Update with your actual path
output_path = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\output_duplicated'  # Update with your actual path
#
model, device = load_model(checkpoint_path)
evaluate_model(test_path, output_path, model, processor, device)