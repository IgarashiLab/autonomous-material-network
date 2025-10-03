# Run the optimization system simultaneously
import os
import subprocess
import argparse
import shutil
from ruamel.yaml import YAML
import glob

def add_suffix_to_directory(directory_path, suffix):
    # Remove the trailing '/' from the directory path
    directory_path = directory_path.rstrip('/')
    
    # Separate the directory name and path
    dir_name = os.path.basename(directory_path)
    parent_path = os.path.dirname(directory_path)
    
    # Create a new directory name (add suffix)
    new_dir_name = f"{dir_name}{suffix}"
    
    # Create a new full path
    new_path = os.path.join(parent_path, new_dir_name)
    
    return new_path

def run_subprocess(filename_stdout, filename_stderr, directory):
    current_dir = os.getcwd()  # Save the current directory
    try:
        os.chdir(directory)  # Change to the specified directory
        return subprocess.Popen(
            ["python3","-u", "run_networkBO.py"],
            stdout=open(filename_stdout, 'w'),
            stderr=open(filename_stderr, 'w'),
        )
    finally:
        os.chdir(current_dir)  # Return to the original directory

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run networkBO multiple times.")
    parser.add_argument('-p', '--network_path', type=str, required=True, help='Path to the network to execute')
    parser.add_argument('-n', '--run_num', type=int, default=10, help='Number of executions')

    # Parse arguments
    args = parser.parse_args()

    network_path = args.network_path
    run_num = args.run_num
    filename_stdout="stdout"
    filename_stderr="stderr"
    result_dir=f"result_{network_path}"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    system_name_list = ["BO_system_CT", "BO_system_MM", "BO_system_SP"]
    for system_name in system_name_list:
        if not os.path.exists(os.path.join(result_dir,system_name)):
            os.mkdir(os.path.join(result_dir,system_name))


    yaml=YAML()
    yaml.preserve_quotes=True
    # Create a temporary system for each seed
    for seed in range(run_num):
        network_path_this=add_suffix_to_directory(network_path,f"_{seed}")
        if not os.path.exists(network_path_this):
            shutil.copytree(network_path, network_path_this)
        open(os.path.join(network_path_this,filename_stdout), 'w')
        open(os.path.join(network_path_this,filename_stderr), 'w')
        # Set the seed in setting.yaml
        with open(os.path.join(network_path_this, "setting.yaml")) as f:
            current_yaml=yaml.load(f)
        current_yaml["common"]["seed"]=seed
        with open(os.path.join(network_path_this, "setting.yaml"), "w") as f:
            yaml.dump(current_yaml, f)


    # Run asynchronously
    process_list = [run_subprocess(filename_stdout, filename_stderr, add_suffix_to_directory(network_path,f"_{seed}")) for seed in range(run_num)]

    # Wait for all processes to complete
    for process in process_list:
        process.wait()

    print("All subprocesses have completed.")

    for seed in range(run_num):

        # Save optimization results for each system
        for system_name in system_name_list:
            shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),system_name,"observed_target.csv"), os.path.join(result_dir,system_name,f"observed_target{seed}.csv"))
            shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),system_name,"unobserved_prediction.csv"), os.path.join(result_dir,system_name,f"unobserved_prediction{seed}.csv"))
            
            # Save new true_estimate_{4-digit number}.csv files
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
            
            # Save new true_estimate_train_{4-digit number}.csv files
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_train_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate_train_",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
            
            # Save new true_estimate_test_{4-digit number}.csv files
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_test_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate_test_",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
            
            # Save new true_estimate_std_{4-digit number}.csv files
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_std_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate_std_",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
        
        
        # Save stdout and stderr
        shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),"stdout"),os.path.join(result_dir,f"stdout{seed}"))
        shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),"stderr"),os.path.join(result_dir,f"stderr{seed}"))

        shutil.rmtree(add_suffix_to_directory(network_path,f"_{seed}"))


if __name__ == '__main__':
    main()