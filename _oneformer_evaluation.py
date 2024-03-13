from transformers import AutoModelForUniversalSegmentation
from _oneformer_custom_data import processor
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import os

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


def iou_score(output, target):
    """
    Compute the Intersection over Union (IoU) score.
    """
    intersection = (output & target).sum()
    union = (output | target).sum()
    if union == 0:
        return float('nan')  # Avoid division by zero
    else:
        return intersection / union


def evaluate_model(images_dir, masks_dir, model, processor, device, num_classes=4):
    class_iou_scores = {class_id: [] for class_id in range(num_classes)}  # Initialize IoU scores for each class

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    for image_file in tqdm(image_files, desc="Evaluating model"):
        base_name = image_file.split('.')[0]  # Remove the file extension to get the base name
        image_path = os.path.join(images_dir, image_file)
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(masks_dir, mask_name)

        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        semantic_segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[(256, 256)])[0]
        prediction = (semantic_segmentation > 0).cpu().numpy().astype(np.uint8)

        # Load the ground truth mask and convert to binary format
        gt_mask = np.array(Image.open(mask_path))

        # For each class, calculate IoU and update class_iou_scores
        for class_id in range(num_classes):
            pred_mask = (prediction == class_id).astype(np.uint8)
            true_mask = (gt_mask == class_id).astype(np.uint8)
            iou = iou_score(pred_mask, true_mask)
            if not np.isnan(iou):
                class_iou_scores[class_id].append(iou)

    # Compute mAP50 for each class and the average
    average_precisions = []
    for class_id, ious in class_iou_scores.items():
        tp = sum(iou >= 0.5 for iou in ious)
        precision = tp / len(ious) if ious else 0
        average_precisions.append(precision)
        print(f"Class {class_id} Precision: {precision:.4f}")

    mean_ap = np.nanmean(average_precisions)
    print(f"Mean Average Precision (mAP@0.5): {mean_ap:.4f}")


if __name__ == '__main__':
    # Example usage
    images_dir = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\APPLE-TRAIN-YOLO\images'  # Update with your actual path
    masks_dir = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\APPLE-TRAIN-YOLO\masks'  # Update with your actual path
    checkpoint_path = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\ADS_oneformer\one_former\train-final-6\final_model.pt'  # Update with your actual path

    model, device = load_model(checkpoint_path)
    evaluate_model(images_dir, masks_dir, model, processor, device)