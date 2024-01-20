import matplotlib.pyplot as plt
import glob
import re
import os
import pandas as pd
from model_handler import ModelHandler
import argparse


def compare_experiments(test_data_path):
    # Collect all the experiment directories in the current directory
    exp_dirs = [dir for dir in glob.glob('YOLOv8-ADL*') if os.path.isdir(dir)]

    max_points = 0
    data = {}

    # First loop to find max_points and collect data
    for exp_path in exp_dirs:
        weight_files = sorted(
            glob.glob(os.path.join(exp_path, 'runs', 'segment', 'active-learning_*', 'weights', 'best.pt')),
            key=lambda x: int(re.findall(r'\d+', x)[0]))

        model_handler = ModelHandler(None, None)

        excel_file = f'{exp_path}_scores.xlsx'

        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
            scores = df['mAP50'].tolist()
        else:
            scores = [model_handler.val(test_data_path).box.map50 for model_handler.model_path in weight_files]
            df = pd.DataFrame(scores, columns=['mAP50']).round(4)
            df.to_excel(excel_file, index=False)

        num_points = len(scores)
        max_points = max(max_points, num_points)

        data[exp_path] = scores

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Second loop to generate the plots
    for exp_path, scores in data.items():
        num_points = len(scores)

        # Compute the 'x' values for each experiment, spreading the points out over the range 0 to max_points-1
        x_values = [i * (max_points - 1) / (num_points - 1) for i in range(num_points)]

        plt.plot(x_values, scores, 'o-', label=f'{os.path.basename(exp_path)}')

        # plt.text(x_values[-1], scores[-1], f'iter{len(scores)}: {scores[-1]:.3f}', ha='center', va='top', fontsize=8)

    plt.xlabel('Run Number')
    plt.ylabel('mAP@0.5')
    plt.title('Comparison of mAP50 Scores Across Different Runs(PV-Tomato Dataset)')
    plt.legend()
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

    plt.savefig('comparison_plot.png', dpi=300)  # Save before show
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare multiple experiments.')
    parser.add_argument('test_data_path', type=str, help='Path to test dataset.')

    args = parser.parse_args()
    compare_experiments(args.test_data_path)

# python test_graph_tomato.py datasets\2_tomato-test\data.yaml