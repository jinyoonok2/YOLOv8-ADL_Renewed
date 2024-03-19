from ultralytics import YOLO
import time
import os
import random

model = YOLO(r"C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\apple_segment\ALL\apple-yolo-loop4\weights\best.pt")
images_dir = r'C:\Jinyoon Projects\datasets\PlantVillage_apple_mask\APPLE-TEST-YOLO\images'

# Get all image files from the directory
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Ensure there are at least 100 images
if len(image_files) < 100:
    print(f"Warning: The directory has only {len(image_files)} images, which is less than 100.")
    sample_images = image_files  # Use all available images if less than 100
else:
    # Randomly select 100 images
    sample_images = random.sample(image_files, 100)

total_inference_time = 0

# Perform inference on each of the 100 randomly selected images
for image_file in sample_images:
    image_path = os.path.join(images_dir, image_file)
    start_time = time.time()
    results = model.predict(image_path)
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time

average_inference_time = total_inference_time / len(sample_images)
print(f"Total inference time for 100 sample images: {total_inference_time:.4f} seconds")
print(f"Average inference time per image: {average_inference_time:.4f} seconds")
