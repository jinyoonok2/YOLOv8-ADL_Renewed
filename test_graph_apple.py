import matplotlib.pyplot as plt
import glob
import re
from model_handler import ModelHandler
import argparse
import os
import pandas as pd


def compare_experiments(exp_path1, exp_path2, test_data_path, manual_path=None):
    # Collect all the weight files in the experiment folders
    weight_files1 = sorted(
        glob.glob(os.path.join(exp_path1, 'runs', 'segment', 'active-learning_*', 'weights', 'best.pt')),
        key=lambda x: int(re.findall(r'\d+', x)[0]))
    weight_files2 = sorted(
        glob.glob(os.path.join(exp_path2, 'runs', 'segment', 'active-learning_*', 'weights', 'best.pt')),
        key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Create a model handler instance
    model_handler1 = ModelHandler(None, None)
    model_handler2 = ModelHandler(None, None)

    excel_file1 = f'{exp_path1}_scores.xlsx'
    excel_file2 = f'{exp_path2}_scores.xlsx'

    # Check if the excel files exist
    if os.path.exists(excel_file1):
        df1 = pd.read_excel(excel_file1)
        scores1 = df1['mAP50'].tolist()
    else:
        # Evaluate each model on the test set and collect the mAP50 scores
        scores1 = [model_handler1.val(test_data_path).box.map50 for model_handler1.model_path in weight_files1]
        # For the first experiment
        df1 = pd.DataFrame(scores1, columns=['mAP50']).round(4)
        df1.to_excel(excel_file1, index=False)

    if os.path.exists(excel_file2):
        df2 = pd.read_excel(excel_file2)
        scores2 = df2['mAP50'].tolist()
    else:
        scores2 = [model_handler2.val(test_data_path).box.map50 for model_handler2.model_path in weight_files2]
        # For the second experiment
        df2 = pd.DataFrame(scores2, columns=['mAP50']).round(4)
        df2.to_excel(excel_file2, index=False)

    # Determine the number of points in each experiment and the maximum number
    num_points1 = len(scores1)
    num_points2 = len(scores2)
    max_points = max(num_points1, num_points2)

    # Compute the 'x' values for each experiment, spreading the points out over the range 0 to max_points-1
    x_values1 = [i * (max_points - 1) / (num_points1 - 1) for i in range(num_points1)]
    x_values2 = [i * (max_points - 1) / (num_points2 - 1) for i in range(num_points2)]

    # Plot the results
    plt.figure(figsize=(10, 5))

    # Use the 'x' values computed above instead of the default range(0, len(scores))
    plt.plot(x_values1, scores1, 'o-', label=f'YOLOv8-ADL(λ=0.1)')
    plt.plot(x_values2, scores2, 'o-', label=f'YOLOv8-ADL(λ=0.2)')
    # plt.plot(x_values1, scores1, 'o-', label=f'10% Lambda Tomato')
    # plt.plot(x_values2, scores2, 'o-', label=f'20% Lambda Tomato')

    # Add the annotation for the last point of each experiment, adjust va for vertical alignment
    plt.text(x_values1[-1], scores1[-1], f'iter{len(scores1)}: {scores1[-1]:.3f}', ha='center', va='top', fontsize=8)
    plt.text(x_values2[-1], scores2[-1], f'iter{len(scores2)}: {scores2[-1]:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Run Number')
    plt.ylabel('mAP@0.5')
    plt.title('Comparison of mAP50 Scores Across Different Runs(PV-Tomato Dataset)')
    plt.legend()

    # Add grid
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

    if manual_path:
        excel_file3 = f'{manual_path}_score.xlsx'
        if os.path.exists(excel_file3):
            df3 = pd.read_excel(excel_file3)
            manual_score = df3['mAP50'].tolist()[0]
        else:
            model_handler3 = ModelHandler(None, None)
            model_handler3.model_path = manual_path
            manual_score = model_handler3.val(test_data_path).box.map50

            df3 = pd.DataFrame([manual_score], columns=['mAP50']).round(4)
            df3.to_excel(excel_file3, index=False)

        # Plot the manual score as a horizontal line
        plt.hlines(manual_score, xmin=0, xmax=max_points - 1, colors='g', linestyles='--', label=f'YOLOv8 (manual)')

    plt.legend()
    plt.savefig('comparison_plot.png', dpi=300)  # Save before show
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two experiments.')
    parser.add_argument('test_data_path', type=str, help='Path to test dataset.')
    parser.add_argument('exp_path1', type=str, help='Path to the first experiment.')
    parser.add_argument('exp_path2', type=str, help='Path to the second experiment.')
    parser.add_argument('manual_path', type=str, help='Path to the manually specified model.', nargs='?', default=None)

    args = parser.parse_args()
    compare_experiments(args.exp_path1, args.exp_path2, args.test_data_path, args.manual_path)


# python test_graph_apple.py datasets\2_tomato-test\data.yaml YOLOv8+ADL_λ=0.1 YOLOv8+ADL_λ=0.2
# python test_graph_apple.py datasets\2_apple-test\data.yaml YOLOv8+ADL_λ=0.1 YOLOv8+ADL_λ=0.2 apple-manual/weights/best.pt