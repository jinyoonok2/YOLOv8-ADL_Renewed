from ultralytics import YOLO
from data_preprocess.yolo_txt.__dataset_path import TEST_APPLE_PATH, TEST_TOMATO_PATH, TEST_HAM_PATH, TEST_APPLE_MODEL, TEST_TOMATO_MODEL, TEST_HAM_MODEL
import os

import pandas as pd
import csv
import matplotlib.pyplot as plt

apple_segment = ['ALL', 'BOTTOM2', 'TOP1', 'TOP2']
tomato_segment = ['ALL', 'ALL_DUPLICATED']
ham_segment = ['ALL']

apple_model_name = 'apple-yolo-loop'
tomato_model_name = 'apple-yolo-loop'
ham_model_name = 'apple-yolo-loop'
weight_path = r'weights\best.pt'


def save_metrics_to_csv(exp_full_name, loop_name, map50, csv_dir):
    # Ensure the directory for the CSV file exists
    os.makedirs(csv_dir, exist_ok=True)

    # Define CSV file path
    csv_file_path = os.path.join(csv_dir, f'{exp_full_name}.csv')

    # Check if the CSV file exists, create if not
    if os.path.exists(csv_file_path):
        map_scores_df = pd.read_csv(csv_file_path)
    else:
        map_scores_df = pd.DataFrame(columns=['loop_name', 'map50'])

    # Append the new mAP score
    new_row = pd.DataFrame({'loop_name': [loop_name], 'map50': [map50]})
    map_scores_df = pd.concat([map_scores_df, new_row], ignore_index=True)

    # Save the DataFrame to CSV
    map_scores_df.to_csv(csv_file_path, index=False)
    print(f"Saved mAP score for {exp_full_name}: loop {loop_name}")


def plot_and_save_graphs(exp_full_name, csv_dir):
    # Read the CSV file
    csv_file_path = os.path.join(csv_dir, f'{exp_full_name}.csv')
    if not os.path.exists(csv_file_path):
        print(f"No data found for {exp_full_name}. Skipping plot.")
        return

    map_scores_df = pd.read_csv(csv_file_path)

    # Plot
    plt.figure()
    plt.plot(map_scores_df['loop_name'], map_scores_df['map50'], marker='o', linestyle='-')
    plt.title(f'mAP@0.5 for {exp_full_name}')
    plt.xlabel('Loop Name')
    plt.ylabel('mAP@0.5')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(os.path.join(csv_dir, f'{exp_full_name}_mAP_plot.png'))
    plt.close()
    print(f"Saved plot for {exp_full_name}")



# Define a function to process the segments
def process_segment(segment, model_path, test_path, model_name, weight_path, csv_dir):
    for exp_type in segment:
        exp_full_name = f"{model_name}_{exp_type}".lower()

        for loop_name in sorted(os.listdir(os.path.join(model_path, exp_type))):
            full_model_path = os.path.join(model_path, exp_type, loop_name, weight_path)

            # Check if mAP score is already saved
            csv_file_path = os.path.join(csv_dir, f'{exp_full_name}.csv')
            if os.path.exists(csv_file_path):
                map_scores_df = pd.read_csv(csv_file_path)
                if loop_name in map_scores_df['loop_name'].values:
                    print(f"mAP score for {exp_full_name}: loop {loop_name} already exists. Skipping...")
                    continue

            # Run validation and get mAP score
            model = YOLO(full_model_path)
            metrics = model.val(data=test_path, split='test')
            map50 = metrics.box.map50

            # Save the mAP score
            save_metrics_to_csv(exp_full_name, loop_name, map50, csv_dir)

        # After processing all loops for the exp_type, generate the plot
        plot_and_save_graphs(exp_full_name, csv_dir)


if __name__ == '__main__':
    csv_dir = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs'  # Define the path to your CSV directory

    process_segment(apple_segment, TEST_APPLE_MODEL, TEST_APPLE_PATH, 'apple', weight_path, csv_dir)
    process_segment(tomato_segment, TEST_TOMATO_MODEL, TEST_TOMATO_PATH, 'tomato', weight_path, csv_dir)
    process_segment(ham_segment, TEST_HAM_MODEL, TEST_HAM_PATH, 'ham', weight_path, csv_dir)

