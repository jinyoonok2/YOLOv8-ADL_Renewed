import argparse
from _yolo_model_handler import ModelHandler
import torch
import gc
from ultralytics import YOLO

from data_preprocess.yolo_txt.__dataset_path import YAML_PATH, ACTIVE_LEARNING_PATH, INFER_PATH, PROJECT_PATH

def main():


    handler = ModelHandler(data_path=YAML_PATH)

    # Train
    # model_name = handler.train(proj_name=PROJECT_PATH, exp_name='apple-yolo-loop1', num_workers=4)
    # print(model_name)

    # Infer
    handler.model_path = r"C:/Jinyoon Projects/YOLOv8-ADL_Renewed/runs/tomato_segment/tomato-yolo-loop5/weights/best.pt"
    handler.infer(active_path=ACTIVE_LEARNING_PATH, output_path=INFER_PATH)




if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    main()
# python manual_train.py datasets\3_apple-manual\data.yaml yolov8s-seg.pt 10 apple-manual