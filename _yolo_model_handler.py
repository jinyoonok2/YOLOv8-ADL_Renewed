from ultralytics import YOLO
import os
import numpy as np
import torch
import gc
import glob
import heapq
from data_preprocess.yolo_txt.__dataset_path import CLASSES, PROJECT_PATH
import yaml
import pandas as pd
from _yolo_result_distribution import correct_data, split_data, count_images_in_active_learning_path

from pathlib import Path

class PredictionDetails:
    def __init__(self, result_object):
        self.result = result_object
        self.conf = torch.max(self.result.boxes.conf).item()

    def __lt__(self, other):
        return self.conf < other.conf

        # Invert the comparison to make it a max-heap
        # return self.conf > other.conf


class ModelHandler:
    def __init__(self, data_path, model_path='yolov8s-seg.pt'):
        self.data_path = data_path
        self.model_path = model_path
        self.heap_size = {}

    def train(self, proj_name, exp_name, num_workers=8, epoch=20, imgsz=256):
        # train with current model_path, then change the model path to EXP_NAME model.
        model = YOLO(self.model_path)
        model.train(data=self.data_path, project=proj_name, name=exp_name, epochs=epoch,
                    imgsz=imgsz, optimizer='Adam', workers=num_workers, device=0)

        # Update model_path to point to the newly trained model
        model_path = os.path.join(PROJECT_PATH, f"{exp_name}/weights/best.pt")

        torch.cuda.empty_cache()
        gc.collect()
        return model_path

    def infer(self, active_path, output_path, tr=30, batch_size=100):
        # since train model returns path like the following:
        # C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\apple_segment\apple-yolo-loop1/weights/best.pt
        # we need to normalize this path to get the experiment name

        # Normalize the path and convert to a Path object
        normalized_model_path = os.path.normpath(self.model_path)
        path_parts = Path(normalized_model_path).parts

        # 1. Extract experiment name (assuming it's the directory before 'weights')
        experiment_name = [part for part in path_parts if 'yolo-loop' in part][-1]  # Get the last occurrence

        csv_file_path = os.path.join(output_path, 'loop_information.csv')


        # 2. Create or update the CSV file
        if os.path.exists(csv_file_path):
            loop_info_df = pd.read_csv(csv_file_path, index_col=0)
        else:
            loop_info_df = pd.DataFrame()

        # If it's the initial experiment, count the number of images in each class folder
        if experiment_name.endswith('1'):
            for cls in CLASSES:
                cls_path = os.path.join(active_path, cls)
                num_images = len(glob.glob(os.path.join(cls_path, "*.jpg")))
                loop_info_df.loc['initial_images', cls] = num_images

        # Check if the 'initial_images' row exists in loop_info_df
        if 'initial_images' in loop_info_df.index:
            initial_image_counts = loop_info_df.loc['initial_images']
        else:
            initial_image_counts = {}

        # Initialize the label count for this experiment
        if experiment_name not in loop_info_df.index:
            loop_info_df.loc[experiment_name] = 0  # Initialize the whole row to 0 if the experiment is new

        # Initialize the model
        model = YOLO(self.model_path)

        # Initialize the heap for each plant type
        heaps = {cls: [] for cls in CLASSES}

        for cls in CLASSES:
            active_cls_path = os.path.join(active_path, cls)
            output_dir = os.path.join(output_path, cls)
            os.makedirs(output_dir, exist_ok=True)

            img_files = glob.glob(os.path.join(active_cls_path, "*.jpg"))  # Assuming images are in .jpg format

            # Calculate the heap size based on the 'tr' percentage
            if cls in initial_image_counts:
                # Use the initial image count from loop_info_df if available
                self.heap_size[cls] = int(initial_image_counts[cls] * (tr / 100))

            # Calculate the heap size based on the 'tr' percentage only once, at the start
            elif cls not in self.heap_size:
                self.heap_size[cls] = int(len(img_files) * (tr / 100))

            for i in range(0, len(img_files), batch_size):  # we're loading and processing them in batches
                batch_files = img_files[i:i + batch_size]
                results = model.predict(batch_files, device=0, conf=0.5)

                # instead of directly using img_files, you can use result.path for original image path of the prediction
                for result in results:
                    if result.masks is not None:
                        prediction_detail = PredictionDetails(result_object=result)

                        # If the heap is smaller than the heap size, just add the element
                        if len(heaps[cls]) < self.heap_size[cls]:
                            heapq.heappush(heaps[cls], prediction_detail)
                        else:
                            # Push the new item on the heap and then pop the smallest item from the heap
                            heapq.heappushpop(heaps[cls], prediction_detail)

            torch.cuda.empty_cache()  # Empty CUDA cache after processing each batch
            gc.collect()

        # Now write out the top 'tr' percentage of images for each plant type
        for cls, heap in heaps.items():
            loop_info_df.at[experiment_name, cls] = len(heap)
            for prediction_details in heap:
                # Retrieve the image file path from the result
                img_path = prediction_details.result.path
                img_filename = os.path.basename(img_path)  # Get the filename with extension from the path
                base_filename = os.path.splitext(img_filename)[0]  # Remove the extension to get the base filename

                # Prepare the directory for output
                output_dir = os.path.join(output_path, str(cls))
                os.makedirs(output_dir, exist_ok=True)
                output_txt_file = os.path.join(output_dir, base_filename + ".txt")  # Construct the txt file path

                # Use the save_txt method to save the label file
                prediction_details.result.save_txt(txt_file=output_txt_file, save_conf=False)

        # Save the updated DataFrame to the CSV file
        loop_info_df.to_csv(csv_file_path)
        self.post_process_data()

        # Call the function to count images in ACTIVE_LEARNING_PATH
        remaining_image_count = count_images_in_active_learning_path()

        return remaining_image_count

    def post_process_data(self):
        print("Starting data correction...")
        correct_data(self.model_path)
        print("Data correction completed.")
        print("Starting data splitting...")
        split_data()
        print("Data splitting completed.")

    def val(self, TEST_DATA_PATH):
        model = YOLO(self.model_path)
        metrics = model.val(data=TEST_DATA_PATH, device=0, split='test')
        torch.cuda.empty_cache()
        gc.collect()
        return metrics
