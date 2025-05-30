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


# systemが最大化する目的変数を指定し、全件データをsystem内に設置
# ensemble_predictionのmeanとdeviationを設定
def set_obj_data(system_path, obj_name):
    obj_dict = {"MM":0,"SP":2,"CT":6}
    obj_index = obj_dict[obj_name]

    B2_full = pd.read_csv("dataset/B2_20220520_full.csv",header=None)
    B2_target = B2_full.iloc[:,obj_index]

    # systemのtarget.csvを更新
    B2_target.to_csv(os.path.join(system_path,"complete_B2_target.csv"), index=False, header=False)

    file_list = glob.glob(os.path.join(system_path,"*/_ensemble_prediction*.ini"))

    print("\n上書きしたもの")
    print("ensemble_.ini list : ", file_list)

    for file_name in file_list:

        config = configparser.ConfigParser()
        config.read(file_name)

        # 'RUN'セクションの'output_mean'を更新
        # config.set('DEFAULT', 'output_mean', f'[{B2_target.mean().iloc[0]:.6f}, {B2_target.mean().iloc[0]:.6f}]')
        # config.set('DEFAULT', 'output_deviation', f'[{B2_target.std().iloc[0]:.6f}, {B2_target.std().iloc[0]:.6f}]')
        config.set('DEFAULT', 'output_mean', f'[{B2_target.mean():.6f}, {B2_target.mean():.6f}]')
        config.set('DEFAULT', 'output_deviation', f'[{B2_target.std():.6f}, {B2_target.std():.6f}]')

        # 更新した設定をINIファイルに書き込む
        with open(file_name, 'w') as configfile:
            config.write(configfile)
    
    print('output_mean', f'[{B2_target.mean():.6f}, {B2_target.mean():.6f}]')
    print('output_deviation', f'[{B2_target.std():.6f}, {B2_target.std():.6f}]')

# 前回の一時ファイルを削除する
def delete_file():
    print("\n削除したファイル")
    # パターンを指定
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

        # カレントディレクトリを取得
        current_directory = os.getcwd()

        regex = re.compile(pattern)
        for root, dirs, files in os.walk(current_directory):
            for file in files:
                if regex.match(file):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"削除されたファイル: {file_path}")
                    except OSError as e:
                        print(f"エラー: {file_path} を削除できませんでした - {e}")
    
    print("")


def main():
    # 引数のパーサーを設定
    parser = argparse.ArgumentParser(description="システムを初期化します。")
    parser.add_argument('-s', '--system_path', required=True,  help='処理目的のsystemのPATH')
    parser.add_argument('-o', '--obj_name', required=True, choices=["CT","MM","SP"], help='最適化目的のパラメータ')
    parser.add_argument('-n', '--observed_num', type=int, default=10, help='既知データの個数')
    parser.add_argument('-r', '--random_seed', type=int, default=-1, help='ランダムシード値')
    parser.add_argument('-f', '--comp_input_feature_name', default='complete_input_features_A.csv', help='利用する特徴量セット')

    # 引数を解析
    args = parser.parse_args()

    system_path = args.system_path
    obj_name = args.obj_name
    observed_num = args.observed_num
    random_seed = args.random_seed
    comp_input_feature_name = args.comp_input_feature_name

    # system_pathを表示
    print(os.path.join(os.getcwd(),system_path))

    # 不要ファイルの削除
    delete_file()

    # complete_input_featuresの初期化
    source_file = os.path.join('dataset', comp_input_feature_name)
    destination_file = os.path.join(system_path, 'complete_input_features.csv')
    shutil.copy(source_file, destination_file)

    # 目的プロパティ値のセット
    set_obj_data(system_path, obj_name)

    # system内のデータセットを初期化
    os.chdir(system_path)
    # print(os.getcwd())

    # 既知データの数, シャッフルするか否か, seed
    if random_seed == -1:
        subprocess.run(['python3', 'dataset_initialize.py', '-o', str(observed_num)])
    else:
        subprocess.run(['python3', 'dataset_initialize.py', '-o', str(observed_num), '-rs', str(random_seed)])


if __name__ == '__main__':
    main()


