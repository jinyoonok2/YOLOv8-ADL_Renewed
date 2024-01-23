import argparse
from _yolo_model_handler import ModelHandler
import torch
import gc
from ultralytics import YOLO

from data_preprocess.yolo_txt.__dataset_path import YAML_PATH, ACTIVE_LEARNING_PATH, INFER_PATH, PROJECT_PATH

def main():

    # # Initialize the ModelHandler with the paths
    handler = ModelHandler(data_path=YAML_PATH)

    # Call the train method
    model_name = handler.train(proj_name=PROJECT_PATH, exp_name='apple-yolo-loop1')
    # runs/segment/apple-yolo-loop1/weights/best.pt

    handler.model_path = model_name
    handler.infer(active_path=ACTIVE_LEARNING_PATH, output_path=INFER_PATH)




if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    main()
# python manual_train.py datasets\3_apple-manual\data.yaml yolov8s-seg.pt 10 apple-manual