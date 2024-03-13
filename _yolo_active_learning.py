from _yolo_model_handler import ModelHandler
from data_preprocess.yolo_txt._dataset_path import YAML_PATH, ACTIVE_LEARNING_PATH, INFER_PATH, PROJECT_PATH

def main():
    # Initialize the ModelHandler with the paths
    handler = ModelHandler(data_path=YAML_PATH)
    base_model_path = 'yolov8s-seg.pt'  # Initial model path
    handler.model_path = base_model_path

    exp_number = 1 # default: 1
    previous_image_count = -1
    current_image_count = 0
    workers = 4

    while True:
        # Define the experiment name
        # exp_name = f'apple-yolo-loop{exp_number}'
        # exp_name = f'tomato-yolo-loop{exp_number}'
        exp_name = f'ham-yolo-loop{exp_number}'

        # Train the model
        model_name = handler.train(proj_name=PROJECT_PATH, exp_name=exp_name, num_workers=workers)
        # Model path example: 'runs/segment/apple-yolo-loop1/weights/best.pt'

        # Set the model path for inference
        handler.model_path = model_name


        # Perform inference
        current_image_count = handler.infer(active_path=ACTIVE_LEARNING_PATH, output_path=INFER_PATH)

        # Check for loop termination condition
        if current_image_count == previous_image_count or current_image_count == 0:
            print("Active learning loop has converged or no images are left.")
            break

        # Reset model path to the base model for the next training
        handler.model_path = base_model_path

        # Update previous image count and increment experiment number
        previous_image_count = current_image_count
        exp_number += 1

if __name__ == "__main__":
    main()