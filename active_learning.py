import os
import yaml
import argparse
import glob
import shutil
from data_preprocessing import split_data, correct_data, recall_existing_data, generate_yaml
from model_handler import ModelHandler
from ultralytics import YOLO


class ActiveLearningLoop:
    def __init__(self, model_handler, train_data_path, train_active_path, tr, imgsz, epoch, initial_model_path):
        self.model_handler = model_handler
        self.train_data_path = train_data_path  # Pointing to the training data folder
        self.tr = tr
        self.imgsz = imgsz
        self.epoch = epoch
        self.initial_model_path = initial_model_path  # Save the initial model path
        self.label_map = None
        self.unannotated_path = train_active_path  # Initially, unannotated_path is same as train_active_path

    def active_learning(self):
        print("starting active learning")
        exp_num = 0
        prev_counts = None  # this will keep track of the previous counts of images in each class

        while True:
            all_files = glob.glob(os.path.join(self.unannotated_path, '**', '*'), recursive=True)
            image_files = [f for f in all_files if os.path.splitext(f)[1] in ['.jpg']]

            # calculate current counts for each class
            cur_counts = {}
            for image_file in image_files:
                class_name = os.path.basename(os.path.dirname(image_file))
                cur_counts[class_name] = cur_counts.get(class_name, 0) + 1

            # if there are no more unannotated images or the counts didn't change, break the loop
            if not image_files or cur_counts == prev_counts:
                break

            print(f'Loop {exp_num}: Found {len(image_files)} unannotated images')

            # Training the model
            exp_name = f'active-learning_{exp_num}'
            self.model_handler.train(exp_name=exp_name, epoch=self.epoch, imgsz=self.imgsz)
            if self.label_map is None:
                self.label_map = YOLO(self.model_handler.model_path).model.names

            # Predicting the labels
            model_path = f'runs/segment/{exp_name}/weights/best.pt'
            self.model_handler.model_path = model_path  # updating the model path to the trained model
            label_path = f'outputs/{exp_name}'
            os.makedirs(label_path, exist_ok=True)
            self.model_handler.infer(img_path=self.unannotated_path, output_path=label_path, label_map=self.label_map,
                                     tr=self.tr)

            # Reset the model path to the initial model path for the next loop
            self.model_handler.model_path = self.initial_model_path

            # Data preprocessing
            correct_data(LABEL_PATH=label_path, label_map=self.label_map)
            split_data(LABEL_PATH=label_path, IMG_PATH=self.unannotated_path, label_map=self.label_map)
            recall_existing_data(prev_data_path=self.train_data_path, label_path=label_path, label_map=self.label_map)
            generate_yaml(label_path=label_path, label_map=self.label_map)

            # Updating paths
            self.train_data_path = label_path
            self.model_handler.data_path = os.path.join(self.train_data_path, 'data.yaml')
            self.unannotated_path = os.path.join(label_path, 'unannotated')  # updated to the new unannotated path

            # after the end of the loop, save the current counts as the previous counts for the next iteration
            prev_counts = cur_counts

            exp_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Learning Loop')
    parser.add_argument('train_data_path', type=str, help='Initial training data path')
    parser.add_argument('train_active_path', type=str, help='Path of dataset with raw images')
    parser.add_argument('train_model', type=str, help='Model to use for training')
    parser.add_argument('tr', type=int, help='Training rate')
    parser.add_argument('imgsz', type=int, help='Size of images')
    parser.add_argument('epoch', type=int, help='Number of epochs')
    args = parser.parse_args()

    initial_model_path = args.train_model  # Save the initial model path
    model_handler = ModelHandler(os.path.join(args.train_data_path, "data.yaml"), args.train_model)
    al = ActiveLearningLoop(model_handler, args.train_data_path, args.train_active_path, args.tr, args.imgsz, args.epoch, initial_model_path)
    al.active_learning()

# python active_learning.py datasets\0_tomato-train-25 datasets\1_tomato-active-learning yolov8s-seg.pt 20 256 10
# python active_learning.py datasets\0_tomato-train-25 datasets\1_tomato-active-learning yolov8s-seg.pt 10 256 10

# python active_learning.py datasets\0_tomato-train-50 datasets\1_tomato-active-learning yolov8s-seg.pt 20 256 10
# python active_learning.py datasets\0_tomato-train-50 datasets\1_tomato-active-learning yolov8s-seg.pt 10 256 10

# python active_learning.py datasets\0_tomato-train-100 datasets\1_tomato-active-learning yolov8s-seg.pt 20 256 10
# python active_learning.py datasets\0_tomato-train-100 datasets\1_tomato-active-learning yolov8s-seg.pt 10 256 10

# python active_learning.py datasets\0_apple-train-test-100 datasets\1_apple-active-learning yolov8s-seg.pt 20 256 10
# python active_learning.py datasets\0_apple-train-test-100 datasets\1_apple-active-learning yolov8s-seg.pt 10 256 10