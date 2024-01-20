from ultralytics import YOLO

import os
import numpy as np
import torch
import gc
import glob
import heapq

class ImageDetails:
    def __init__(self, img_file, cls, conf, masks):
        self.img_file = img_file
        self.cls = cls
        self.conf = torch.max(conf).item()
        self.masks = masks

    def __lt__(self, other):
        return self.conf < other.conf

class ModelHandler:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.heap_size = {}

    def train(self, epoch, exp_name, imgsz=256):
        # train with current model_path, then change the model path to EXP_NAME model.
        model = YOLO(self.model_path)
        model.train(data=self.data_path, epochs=epoch, name=exp_name, imgsz=imgsz, device=0)

        # Update model_path to point to the newly trained model
        model_path = f"runs/segment/{exp_name}/weights/best.pt"
        self.model_path = os.path.join(os.getcwd(), model_path)

        torch.cuda.empty_cache()
        gc.collect()

    def infer(self, img_path, output_path, label_map, tr, batch_size=100):
        # Extract the plant names from the label map
        plants = list(label_map.values())
        # Initialize the model
        model = YOLO(self.model_path)

        # Initialize the heap for each plant type
        heaps = {ptype: [] for ptype in plants}

        for ptype in plants:
            plant_path = os.path.join(img_path, ptype)
            output_dir = os.path.join(output_path, ptype)
            os.makedirs(output_dir, exist_ok=True)

            img_files = glob.glob(os.path.join(plant_path, "*.jpg"))  # Assuming images are in .jpg format

            # Calculate the heap size based on the 'tr' percentage only once, at the start
            if ptype not in self.heap_size:
                self.heap_size[ptype] = int(len(img_files) * (tr / 100))

            for i in range(0, len(img_files), batch_size):  # we're loading and processing them in batches
                batch_files = img_files[i:i + batch_size]
                results = model.predict(batch_files, device=0)

                for result, img_file in zip(results, batch_files):
                    if result.masks is not None:
                        cls = result.boxes.cls
                        conf = result.boxes.conf
                        masks = result.masks.xyn

                        # Print the conf tensor
                        heaps[ptype].append(ImageDetails(img_file, cls, conf, masks))

                        # Maintain the heap size to the calculated size
                        if len(heaps[ptype]) > self.heap_size[ptype]:
                            heapq.heappop(heaps[ptype])

                torch.cuda.empty_cache()  # Empty CUDA cache after processing each batch
            gc.collect()

        # Now write out the top 'tr' percentage of images for each plant type
        for ptype, heap in heaps.items():
            for image_details in heap:
                masks = image_details.masks
                cls_value = int(image_details.cls.cpu().numpy()[0])

                # Format output and write to file
                masks_flattened = np.concatenate(masks).flatten()
                output_str = f"{cls_value} " + " ".join(map(str, masks_flattened))

                # If masks are not in the expected format, print a message and skip this iteration
                if masks_flattened is None or len(masks_flattened) == 0:
                    print(f"Skipping {image_details.img_file} due to empty or None masks.")
                    continue

                base_filename = os.path.splitext(os.path.basename(image_details.img_file))[0]
                output_dir = os.path.join(output_path, ptype)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, base_filename + ".txt")

                with open(output_file, "w") as f:
                    f.write(output_str)

    def val(self, TEST_DATA_PATH):
        model = YOLO(self.model_path)
        metrics = model.val(data=TEST_DATA_PATH, device=0)
        torch.cuda.empty_cache()
        gc.collect()
        return metrics
