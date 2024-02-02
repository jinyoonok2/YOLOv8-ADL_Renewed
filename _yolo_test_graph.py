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


def collect_and_plot_segment_data(model_name, segment, csv_dir, plot_dir):
    # Prepare the plot directory
    os.makedirs(plot_dir, exist_ok=True)

    # Collect data from each CSV corresponding to the segment type
    segment_data = pd.DataFrame()
    for exp_type in segment:
        exp_full_name = f"{model_name}_{exp_type}".lower()
        csv_file_path = os.path.join(csv_dir, f'{exp_full_name}.csv')
        if os.path.exists(csv_file_path):
            exp_data = pd.read_csv(csv_file_path)
            exp_data['exp_name'] = exp_full_name  # Add column to identify the experiment
            segment_data = pd.concat([segment_data, exp_data], ignore_index=True)

    # If segment_data is empty, there's nothing to plot.
    if segment_data.empty:
        print(f"No data found for any experiments in the segment: {segment}. Skipping plot.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    for exp_name in segment_data['exp_name'].unique():
        exp_data = segment_data[segment_data['exp_name'] == exp_name]
        plt.plot(exp_data['loop_name'], exp_data['map50'], label=exp_name, marker='o')

    # Configure and save the plot
    plt.title(f'mAP@0.5 for {model_name}')
    plt.xlabel('Loop Name')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.ylim(0, 1.0)  # Set the limits of the y-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = f'{model_name}_mAP_plot.png'
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.close()
    print(f"Plot saved as {plot_filename}")




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


if __name__ == '__main__':
    csv_dir = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\csv'  # Define the path to your CSV directory
    plot_dir = r'C:\Jinyoon Projects\YOLOv8-ADL_Renewed\runs\plots'  # Define the path to your plot directory

    # Process segments and save CSVs
    process_segment(apple_segment, TEST_APPLE_MODEL, TEST_APPLE_PATH, 'apple', weight_path, csv_dir)
    process_segment(tomato_segment, TEST_TOMATO_MODEL, TEST_TOMATO_PATH, 'tomato', weight_path, csv_dir)
    process_segment(ham_segment, TEST_HAM_MODEL, TEST_HAM_PATH, 'ham', weight_path, csv_dir)

    # After processing all segments, collect data and plot graphs
    collect_and_plot_segment_data('apple', apple_segment, csv_dir, plot_dir)
    collect_and_plot_segment_data('tomato', tomato_segment, csv_dir, plot_dir)
    collect_and_plot_segment_data('ham', ham_segment, csv_dir, plot_dir)

