import argparse
from model_handler import ModelHandler
import torch
import gc

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model.')

    # Add the arguments
    parser.add_argument('data_path', type=str, help='the path to the data.yaml file')

    parser.add_argument('model_path',type=str, help='the path to the yolov8s-seg.pt model file')

    parser.add_argument('epochs',type=int, help='the number of epochs for training')

    parser.add_argument('exp_name', type=str, help='the experiment name for training')

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the ModelHandler with the paths
    handler = ModelHandler(data_path=args.data_path, model_path=args.model_path)

    # Call the train method
    handler.train(args.epochs, args.exp_name, 256)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    main()
# python manual_train.py datasets\3_apple-manual\data.yaml yolov8s-seg.pt 10 apple-manual