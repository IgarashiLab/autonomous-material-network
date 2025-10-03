import argparse
import os
import shutil
import subprocess
import sys
import pandas as pd
import numpy as np
import configparser
import glob
import re


# Specify the objective variable that the system maximizes and place all data in the system
# Set the mean and deviation for ensemble_prediction
def set_obj_data(system_path, obj_name):
    obj_dict = {"MM":0,"SP":2,"CT":6}
    obj_index = obj_dict[obj_name]

    B2_full = pd.read_csv("dataset/B2_20220520_full.csv",header=None)
    B2_target = B2_full.iloc[:,obj_index]

    # Update the system's target.csv
    B2_target.to_csv(os.path.join(system_path,"complete_B2_target.csv"), index=False, header=False)

    file_list = glob.glob(os.path.join(system_path,"*/_ensemble_prediction*.ini"))

    print("\nOverwritten files")
    print("ensemble_.ini list : ", file_list)

    for file_name in file_list:

        config = configparser.ConfigParser()
        config.read(file_name)

        # Update 'output_mean' in the 'RUN' section
        # config.set('DEFAULT', 'output_mean', f'[{B2_target.mean().iloc[0]:.6f}, {B2_target.mean().iloc[0]:.6f}]')
        # config.set('DEFAULT', 'output_deviation', f'[{B2_target.std().iloc[0]:.6f}, {B2_target.std().iloc[0]:.6f}]')
        config.set('DEFAULT', 'output_mean', f'[{B2_target.mean():.6f}, {B2_target.mean():.6f}]')
        config.set('DEFAULT', 'output_deviation', f'[{B2_target.std():.6f}, {B2_target.std():.6f}]')

        # Write the updated settings to the INI file
        with open(file_name, 'w') as configfile:
            config.write(configfile)
    
    print('output_mean', f'[{B2_target.mean():.6f}, {B2_target.mean():.6f}]')
    print('output_deviation', f'[{B2_target.std():.6f}, {B2_target.std():.6f}]')

# Delete previous temporary files
def delete_file():
    print("\nDeleted files")
    # Specify patterns
    pattern_list=[
        "predicted_\d+_train.csv",
        "predicted_\d+_test.csv",
        "model_.+.h5",
        "unobserved_ensemble_predicted_\d+.csv",
        "Bayesian_\d+.csv",
        "virtual_material_search.results",
        "unobserved_prediction.csv",
        "true_estimate_.+.csv",
    ]

    for pattern in pattern_list:

        # Get the current directory
        current_directory = os.getcwd()

        regex = re.compile(pattern)
        for root, dirs, files in os.walk(current_directory):
            for file in files:
                if regex.match(file):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except OSError as e:
                        print(f"Error: Could not delete {file_path} - {e}")
    
    print("")


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Initialize the system.")
    parser.add_argument('-s', '--system_path', required=True,  help='PATH of the system to process')
    parser.add_argument('-o', '--obj_name', required=True, choices=["CT","MM","SP"], help='Parameter to optimize')
    parser.add_argument('-n', '--observed_num', type=int, default=10, help='Number of known data')
    parser.add_argument('-r', '--random_seed', type=int, default=-1, help='Random seed value')
    parser.add_argument('-f', '--comp_input_feature_name', default='complete_input_features_A.csv', help='Feature set to use')

    # Parse arguments
    args = parser.parse_args()

    system_path = args.system_path
    obj_name = args.obj_name
    observed_num = args.observed_num
    random_seed = args.random_seed
    comp_input_feature_name = args.comp_input_feature_name

    # Display the system_path
    print(os.path.join(os.getcwd(),system_path))

    # Delete unnecessary files
    delete_file()

    # Initialize complete_input_features
    source_file = os.path.join('dataset', comp_input_feature_name)
    destination_file = os.path.join(system_path, 'complete_input_features.csv')
    shutil.copy(source_file, destination_file)

    # Set the objective property value
    set_obj_data(system_path, obj_name)

    # Initialize the dataset in the system
    os.chdir(system_path)
    # print(os.getcwd())

    # Number of known data, whether to shuffle, seed
    if random_seed == -1:
        subprocess.run(['python3', 'dataset_initialize.py', '-o', str(observed_num)])
    else:
        subprocess.run(['python3', 'dataset_initialize.py', '-o', str(observed_num), '-rs', str(random_seed)])


if __name__ == '__main__':
    main()


