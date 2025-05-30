# 最適化システムを同時にrunする
import os
import subprocess
import argparse
import shutil
from ruamel.yaml import YAML
import glob

def add_suffix_to_directory(directory_path, suffix):
    # ディレクトリパスの末尾の '/' を除去
    directory_path = directory_path.rstrip('/')
    
    # ディレクトリ名とパスを分離
    dir_name = os.path.basename(directory_path)
    parent_path = os.path.dirname(directory_path)
    
    # 新しいディレクトリ名を作成（接尾語を追加）
    new_dir_name = f"{dir_name}{suffix}"
    
    # 新しい完全なパスを作成
    new_path = os.path.join(parent_path, new_dir_name)
    
    return new_path

def run_subprocess(filename_stdout, filename_stderr, directory):
    current_dir = os.getcwd()  # 現在のディレクトリを保存
    try:
        os.chdir(directory)  # 指定されたディレクトリに移動
        return subprocess.Popen(
            ["python3","-u", "run_networkBO.py"],
            stdout=open(filename_stdout, 'w'),
            stderr=open(filename_stderr, 'w'),
        )
    finally:
        os.chdir(current_dir)  # 元のディレクトリに戻る

def main():
    # 引数のパーサーを設定
    parser = argparse.ArgumentParser(description="networkBOを複数回実行する。")
    parser.add_argument('-p', '--network_path', type=str, required=True, help='実行するnetwork path')
    parser.add_argument('-n', '--run_num', type=int, default=10, help='実行回数')

    # 引数を解析
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
    # seedごとにsystemを一時的に作る
    for seed in range(run_num):
        network_path_this=add_suffix_to_directory(network_path,f"_{seed}")
        if not os.path.exists(network_path_this):
            shutil.copytree(network_path, network_path_this)
        open(os.path.join(network_path_this,filename_stdout), 'w')
        open(os.path.join(network_path_this,filename_stderr), 'w')
        # setting.yamlにseedを設定する
        with open(os.path.join(network_path_this, "setting.yaml")) as f:
            current_yaml=yaml.load(f)
        current_yaml["common"]["seed"]=seed
        with open(os.path.join(network_path_this, "setting.yaml"), "w") as f:
            yaml.dump(current_yaml, f)


    # 非同期でrunする
    process_list = [run_subprocess(filename_stdout, filename_stderr, add_suffix_to_directory(network_path,f"_{seed}")) for seed in range(run_num)]

    # すべてのプロセスが完了するのを待つ
    for process in process_list:
        process.wait()

    print("すべてのサブプロセスが完了しました。")

    for seed in range(run_num):

        # 各システムの最適化結果を保存
        for system_name in system_name_list:
            shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),system_name,"observed_target.csv"), os.path.join(result_dir,system_name,f"observed_target{seed}.csv"))
            shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),system_name,"unobserved_prediction.csv"), os.path.join(result_dir,system_name,f"unobserved_prediction{seed}.csv"))
            
            # 新たにtrue_estimate_{数字4桁}.csvファイルを保存
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
            
            # 新たにtrue_estimate_{数字4桁}.csvファイルを保存
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_train_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate_train_",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
            
            # 新たにtrue_estimate_{数字4桁}.csvファイルを保存
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_test_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate_test_",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
            
            # 新たにtrue_estimate_{数字4桁}.csvファイルを保存
            true_estimate_files = glob.glob(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"), system_name, "true_estimate_std_*.csv"))
            true_estimate_dir = os.path.join(result_dir, system_name, "true_estimate_std_",f"seed{seed}")
            os.makedirs(true_estimate_dir, exist_ok=True)
            for true_estimate_file in true_estimate_files:
                file_name = os.path.basename(true_estimate_file)
                shutil.copy(true_estimate_file, os.path.join(true_estimate_dir, file_name))
        
        
        # stdoutとstderrを保存
        shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),"stdout"),os.path.join(result_dir,f"stdout{seed}"))
        shutil.copy(os.path.join(add_suffix_to_directory(network_path,f"_{seed}"),"stderr"),os.path.join(result_dir,f"stderr{seed}"))

        shutil.rmtree(add_suffix_to_directory(network_path,f"_{seed}"))


if __name__ == '__main__':
    main()