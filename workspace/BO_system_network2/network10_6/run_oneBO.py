import argparse
import os
import subprocess
import re
import numpy as np
import shutil
import pandas as pd
import matplotlib.pyplot as plt

def extract_train_loss(file_path, mode="val_loss"):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epoch_val_loss_pairs = []
    epoch_pattern = re.compile(r"Epoch\s+(\d+)/\d+")
    if mode=="val_loss":
        loss_pattern = re.compile(r"val_loss:\s+([\d\.]+)")
    elif mode=="loss":
        loss_pattern = re.compile(r"(?<!val_)loss:\s+([\d\.]+)")
    else:
        print("modeの指定がおかしいです。")
        return None

    current_epoch = None

    for line in lines:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        val_loss_match = loss_pattern.search(line)
        if val_loss_match and current_epoch is not None:
            val_loss = float(val_loss_match.group(1))
            epoch_val_loss_pairs.append((current_epoch, val_loss))

    loss_lists=[]
    loss_list=[]
    last_epoch=-1
    for epoch, val_loss in epoch_val_loss_pairs:
        if last_epoch>epoch:
            loss_lists.append(loss_list)
            loss_list=[]
        else:
            loss_list.append(val_loss)
        last_epoch=epoch
    loss_lists.append(loss_list)

    return loss_lists


def main():
    # 引数のパーサーを設定
    parser = argparse.ArgumentParser(description="最適化処理を1回行う")
    parser.add_argument('-s', '--system_path', required=True, help='処理対象のsystem')
    parser.add_argument('-m', '--mode', default='no_transfer', choices=['no_transfer', 'transfer'], help='モデル学習のモード')
    parser.add_argument('--no_BO', action='store_false', help='ベイズ最適化を実行せず、データベースを変更しないようにする。')

    parser.set_defaults(no_BO=True)

    # 引数を解析
    args = parser.parse_args()

    system_path = args.system_path
    mode = args.mode
    no_BO = args.no_BO
    result_filename = 'virtual_material_search.results'

    # システムディレクトリに移動
    os.chdir(system_path)

    # カレントディレクトリ直下のディレクトリのリストを取得
    directories = [d for d in os.listdir('.') if os.path.isdir(d)]
    if mode=="no_transfer":
        directories = ["no_transfer"]

    # 2個同時予測モデルを省く
    pattern = r"transfer_[A-Z]{2}_[A-Z]{2}$"
    union_dirs = [item for item in directories if re.match(pattern, item)]
    directories = [item for item in directories if not re.match(pattern, item)]

    print("処理対象のディレクトリ：",directories)

    # 他のsystemからscratchモデルを元ドメインモデルとして得る
    for dir in directories:
        pattern = r"transfer_(.*)"
        match =re.match(pattern, dir)
        if match:
            model_dir = f"../BO_system_{match.group(1)}/no_transfer/"
            model_pattern=r"model_(\d+)\.h5"
            # modelファイルをコピー
            for filename in os.listdir(model_dir):
                model_match=re.match(model_pattern, filename)
                if model_match:
                    shutil.copy(os.path.join(model_dir,filename), os.path.join(dir,f"model_pre-trained_{model_match.group(1)}.h5"))

    # 同時予測モデルの学習に使えるデータ数を数える
    for dir in union_dirs:
        pattern = r"transfer_(.*)_(.*)"
        match = re.match(pattern, dir)
        if match:
            data_dir1 = f"../BO_system_{match.group(1)}/"
            data_dir2 = f"../BO_system_{match.group(2)}/"
            array1 = np.loadtxt(os.path.join(data_dir1,"observed_input_features.csv"), delimiter=",")
            array2 = np.loadtxt(os.path.join(data_dir2,"observed_input_features.csv"), delimiter=",")
            set1 = set(map(tuple, array1))
            set2 = set(map(tuple, array2))
            union_set=set1.intersection(set2)
            union_array=np.array(list(union_set))
            print(f"{match.group(1)}と{match.group(2)}の共通合金データ数: {union_array.shape[0]}")


    # モデル学習の実行
    for dir in directories:
        # 出力ファイルを初期化
        with open(os.path.join(dir,result_filename), 'w') as f:
            pass
        if dir=="no_transfer":
            subprocess.run(['python', 'train_from_scratch.py', f'{dir}/_train_from_scratch.ini'], stdout=open(os.path.join(dir,result_filename), 'a'))
        else:
            subprocess.run(['python', 'transfer_learning.py', f'{dir}/_transfer_learning.ini'], stdout=open(os.path.join(dir,result_filename), 'a'))

    

    # モデルの選択
    # val_lossの平均が小さいものを選択する
    val_loss_mean_list=[]
    for dir in directories:
        val_loss_list = extract_train_loss(os.path.join(dir, result_filename))
        val_loss_last = [val_loss_list[i][-1] for i in range(len(val_loss_list))]
        val_loss_mean_list.append(np.array(val_loss_last).mean())
    min_index = val_loss_mean_list.index(min(val_loss_mean_list))
    selected_model =directories[min_index]
    print(f"選択したモデル：{selected_model}, val_loss：{val_loss_mean_list[min_index]}")

    # 選択したモデルの学習データに対する予測結果を集約。true_estimate_train{observed_num}.csvとtrue_estimate_val{observed_num}.csvを作成
    import glob
    def parse_list(s):
        return [float(x) for x in s.strip('[]').split()]
    
    obseved_num = pd.read_csv("observed_input_features.csv", header=None).shape[0]
    
    csv_files = glob.glob(f"{selected_model}/predicted_???_train.csv")
    csv_files.sort()

    df_list = []
    for csv_file in csv_files:
        # 0: observed, 1: estimated
        df = pd.read_csv(csv_file, header=None, converters={0: parse_list, 1: parse_list})
        df_list.append(df)

    # dfの要素がリストになっているので、平均値を取る
    for df in df_list:
        df[0] = df[0].apply(lambda x: np.mean(x))
        df[1] = df[1].apply(lambda x: np.mean(x))
    
    # df_listのdfは同じ形状なので、各要素の平均を取り、1つのdfにまとめる
    num_df = len(df_list)
    sum_df = sum(df_list)
    mean_df_train = sum_df / num_df
    mean_df_train.columns = ['observed', 'estimated']

    # 予測結果をcsvに保存
    mean_df_train.to_csv(f"true_estimate_train_{str(obseved_num).zfill(4)}.csv", index=False)

    # valデータについても同様に
    csv_files = glob.glob(f"{selected_model}/predicted_???_test.csv")
    csv_files.sort()

    df_list = []
    for csv_file in csv_files:
        # 0: observed, 1: estimated
        df = pd.read_csv(csv_file, header=None, converters={0: parse_list, 1: parse_list})
        df_list.append(df)
    
    # dfの要素がリストになっているので、平均値を取る
    for df in df_list:
        df[0] = df[0].apply(lambda x: np.mean(x))
        df[1] = df[1].apply(lambda x: np.mean(x))
    
    # df_listのdfは同じ形状なので、各要素の平均を取り、1つのdfにまとめる
    num_df = len(df_list)
    sum_df = sum(df_list)
    mean_df_test = sum_df / num_df
    mean_df_test.columns = ['observed', 'estimated']
    
    # 予測結果をcsvに保存
    mean_df_test.to_csv(f"true_estimate_test_{str(obseved_num).zfill(4)}.csv", index=False)
    

    full_train_data = pd.concat([mean_df_train["observed"], mean_df_test["observed"]], axis=0)

    # 予測時も、学習時と同じmeanとdeviationを使うようにする
    import configparser

    inifilename = f"{selected_model}/_ensemble_prediction.ini"

    config = configparser.ConfigParser()
    config.read(inifilename)

    config.set('DEFAULT', 'output_mean', f'[{full_train_data.mean():.6f}, {full_train_data.mean():.6f}]')
    config.set('DEFAULT', 'output_deviation', f'[{full_train_data.std():.6f}, {full_train_data.std():.6f}]')

    print("DEBUG_set_mean_std:" ,system_path, selected_model, f"{full_train_data.mean():.6f}", f"{full_train_data.std():.6f}")

    # 更新した設定をINIファイルに書き込む
    with open(inifilename, 'w') as configfile:
        config.write(configfile)


    # 選択したモデルで予測とBOを実行する。
    subprocess.run(['python', 'ensemble_prediction.py', f'{selected_model}/_ensemble_prediction.ini'], stdout=open(os.path.join(selected_model,result_filename), 'a'))
    if no_BO:
        subprocess.run(['python', 'Bayesian_optimization.py', f'{selected_model}/_Bayesian_optimization.ini'], stdout=open(os.path.join(selected_model,result_filename), 'a'))

if __name__ == '__main__':
    main()