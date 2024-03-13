import os
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from transformers import AutoModelForUniversalSegmentation
import random
import shutil

from data_preprocess.one_former_mask.dataset_path import CLASSES, ID2LABEL

class ImageClassInferencer:
    def __init__(self, model_checkpoint_path, yaml_path, device='cuda'):
        self.model_checkpoint_path = model_checkpoint_path
        self.yaml_path = yaml_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.classes, self.id2label = self.load_classes()

    def load_model(self):
        model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large",
                                                                  is_training=True)
        checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.model.is_training = False
        model.to(self.device)
        model.eval()
        return model

    def infer(self, input_dir_path, output_dir_path, processor, sample_percentage=0.3):
        # Ensure output directory exists
        os.makedirs(output_dir_path, exist_ok=True)

        for class_name in self.classes:
            class_dir = os.path.join(input_dir_path, class_name)
            output_class_dir = os.path.join(output_dir_path, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            if os.path.isdir(class_dir):
                images = os.listdir(class_dir)
                total_images = len(images)
                sample_size = int(total_images * sample_percentage)

                sampled_images = random.sample(images, sample_size)

                for image_name in sampled_images:
                    image_path = os.path.join(class_dir, image_name)
                    mask_output_path = os.path.join(output_class_dir, image_name)
                    self.process_image(image_path, mask_output_path, processor)

    def process_image(self, image_path, mask_output_path, processor):
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        semantic_segmentation = processor.post_process_semantic_segmentation(outputs)[0]
        mask = self.create_mask(semantic_segmentation)
        mask.save(mask_output_path)
        print(f"Processed and saved mask for {os.path.basename(image_path)}")

    def create_mask(self, segmentation_map):
        mask_np = segmentation_map.cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask_np)
        return mask_image


# Example usage:
if __name__ == "__main__":
    MODEL_CHECKPOINT_PATH = "path_to_model_checkpoint.pt"
    YAML_PATH = "path_to_classes_yaml.yaml"
    INPUT_DIR_PATH = "path_to_input_directory"
    OUTPUT_DIR_PATH = "path_to_output_directory"

    # Assuming `processor` is already defined as per your environment
    inferencer = ImageClassInferencer(MODEL_CHECKPOINT_PATH, YAML_PATH)
    inferencer.infer(INPUT_DIR_PATH, OUTPUT_DIR_PATH, processor)
