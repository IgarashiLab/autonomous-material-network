import configparser
import csv
import random
import datetime
import numpy as np
import heapq
from copy import deepcopy 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error as calculate_mae
from sklearn.metrics import mean_squared_error as calculate_mse
import os


def pencil_shuffle(array, seed):
    # Fisher-Yates shuffle
    random.seed(seed)
    array = list(array)
    empty_array = []
    total_length = len(array)
    for i in range(total_length):
        j = random.randrange(len(array))
        empty_array.append(array.pop(j))
    return empty_array



#=======================================================
#         Read configuration file
#=======================================================
# 引数から読み込む設定ファイル名を決定
import sys

if len(sys.argv) < 2:
    print("エラー：設定ファイル名を引数として入力してください。")
    sys.exit(1)

config_filename = sys.argv[1]
print(f"iniファイル：{config_filename}")

config_ini = configparser.ConfigParser()
config_ini.read(config_filename, encoding='utf-8')

config_ini = configparser.ConfigParser()
config_ini.read(config_filename, encoding='utf-8')



#=======================================================
#         Hyperparameters
#=======================================================

observed_input_features_file_name = str(config_ini.get('RUN', 'observed_input_features_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
unobserved_input_features_file_name = str(config_ini.get('RUN', 'unobserved_input_features_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"


observed_target_file_name = str(config_ini.get('RUN', 'observed_target_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
predicted_target_file_name_prefix = str(config_ini.get('RUN', 'predicted_target_file_name_prefix')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
# NOTE that length of predicted_target == length of unobserved_input_features

number_of_ensemble = int(config_ini.get('RUN', 'number_of_ensemble')) # Default=10
starting_random_seed = int(config_ini.get('RUN', 'random_initial')) # Default=0
length_of_random_seed_text = int(config_ini.get('RUN', 'length_of_random_seed_text')) # Default=3


type_of_optimization = str(config_ini.get('RUN', 'type_of_optimization')) # Default="maximize"
# 'maximize' or 'minimize'
acquisition_function = str(config_ini.get('RUN', 'acquisition_function')) # Default="UCB"
# 'UCB' (Upper Confidence Bound), 'EI' (Expected Improvment), 'PI' (Probability of improvement)

ucb_alpha = float(config_ini.get('RUN', 'ucb_alpha')) # Default=3.0
# confidence interval parameter
small_xi = float(config_ini.get('RUN', 'small_xi')) # Default=0.001
# stability factor for EI and PI

keep_how_many_candidates = int(config_ini.get('RUN', 'keep_how_many_candidates')) # Default=1
# number of top candidates that will be printed out (searched)

suggested_candidates_file_name_prefix = str(config_ini.get('RUN', 'suggested_candidates_file_name_prefix'))
# Full name of this csv file include number of observed data.

do_not_shuffle = eval(config_ini.get('RUN', 'do_not_shuffle')) # Default=False
mix_random_seed = int(config_ini.get('RUN', 'mix_random_seed')) # Default=0
fake_observation = eval(config_ini.get('RUN', 'fake_observation')) # Default=True

complete_input_features_file_name = str(config_ini.get('RUN', 'complete_input_features_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
complete_target_file_name = str(config_ini.get('RUN', 'complete_target_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"


#=======================================================
#         Setup
#=======================================================

np.set_printoptions(threshold=np.inf) # for maintenance
#np.set_printoptions(threshold=np.100) # original setting

if type_of_optimization != "maximize" and  type_of_optimization != "minimize":
    assert False, (
        "Type of optimization must be maximize or minimize.")

if acquisition_function != "UCB" : # and acquisition_function != "EI" and acquisition_function != "PI" :
    assert False, (
        #"Acquisition function must be UCB or EI or PI.")
        "Acquisition function must be UCB. EI and PI are not yet implemented.")

if keep_how_many_candidates < 1:
    assert False, (
        "keep_how_many_candidates must be 1 or greater")


print ("")
print ("Program started")
print ("")
print ("TIME CHECK")
print (datetime.datetime.now())
print ("")



#=======================================================
#         Read data
#=======================================================

with open(observed_input_features_file_name) as f:
    reader = csv.reader(f)
    observed_input_features_csv = [row for row in reader]

with open(unobserved_input_features_file_name) as f:
    reader = csv.reader(f)
    unobserved_input_features_csv = [row for row in reader]

with open(observed_target_file_name) as f:
    reader = csv.reader(f)
    observed_target_csv = [row for row in reader]


observed_target = deepcopy(observed_target_csv)

if len(observed_input_features_csv) != len(observed_target) :
    assert False, (
        "Size of observed_input_features and observed_target must be identical.")
if len(observed_input_features_csv[0]) != len(unobserved_input_features_csv[0]) :
    assert False, (
        "Length of input features (descriptors) must be identical.")
# They are not directly used in this script, but checked to ensure that the suggestions for the next cycle are correct.


number_of_observed = len(observed_input_features_csv)
number_of_unobserved = len(unobserved_input_features_csv)
length_of_descriptor = len(unobserved_input_features_csv[0])

candidates_descriptor = np.zeros((number_of_unobserved, length_of_descriptor), dtype='float32')
candidates_ID = np.zeros((number_of_unobserved, 1), dtype='int')

for mat_ID in range(number_of_unobserved):
    candidates_descriptor[mat_ID] = unobserved_input_features_csv[mat_ID]
    # candidates_ID[mat_ID] = mat_ID + 1 # Starts from 1, not 0
    candidates_ID[mat_ID] = mat_ID # Starts from 0, not 1!!



#=======================================================
#         Read ensemble data
#=======================================================

number_of_output = len(observed_target[0])
candidates_prediction = np.zeros((number_of_unobserved, number_of_ensemble*number_of_output), dtype='float32')

for ensemble_ID in range(number_of_ensemble):
    with open(predicted_target_file_name_prefix + str(ensemble_ID+starting_random_seed).zfill(length_of_random_seed_text) + ".csv" ) as f:
        reader = csv.reader(f)
        predicted_target = [row for row in reader]
    
    for mat_ID in range(number_of_unobserved):
        for out_ID in range(number_of_output):
            candidates_prediction[mat_ID][ensemble_ID*number_of_output+out_ID] = float(predicted_target[mat_ID][out_ID])
            #candidates_prediction[ensemble_ID][mat_ID*number_of_output+out_ID] = float(predicted_target[mat_ID][out_ID])
            # NOTE did not checked yet with 1 output node



#=======================================================
#         Data shuffle with same order
#=======================================================

# if not do_not_shuffle:
#     candidates_prediction = pencil_shuffle(candidates_prediction, mix_random_seed)
#     candidates_descriptor = pencil_shuffle(candidates_descriptor, mix_random_seed)
#     candidates_ID = pencil_shuffle(candidates_ID, mix_random_seed)



#=======================================================
#         Ensemble average and deviation
#=======================================================

print ("==================================================")
print (" You have ", number_of_observed, "observed data.")
print (" You have ", number_of_unobserved, "unobserved data.")
print ("==================================================")
print ("")
print ("TIME CHECK")
print (datetime.datetime.now())

 
candidates_avg = []
candidates_std = []
#candidates_best_ever = 0.0
candidates_acq = []

for mat_ID in range(number_of_unobserved):
    candidates_avg.append(np.average(candidates_prediction[mat_ID]))
    candidates_std.append(np.std(candidates_prediction[mat_ID]))


if type_of_optimization == 'maximize':
    #print (observed_target)
    candidates_best_ever = max(observed_target)[0]
    #print (candidates_best_ever)
else: # 'minimize'
    candidates_best_ever = min(observed_target)[0]



#=======================================================
#         Evaluate acquisition function
#=======================================================

if acquisition_function == 'UCB': # 'UCB' (Upper Confidence Bound), 'EI' (Expected Improvment), 'PI' (Probability of improvement)
    if type_of_optimization == 'minimize':
        ucb_alpha *= -1.0
    
    for mat_ID in range(number_of_unobserved):
        calculated_acq = candidates_avg[mat_ID] + candidates_std[mat_ID] * ucb_alpha
        candidates_acq.append([calculated_acq, int(candidates_ID[mat_ID])])

#elif acquisition_function == 'EI': # 'UCB' (Upper Confidence Bound), 'EI' (Expected Improvment), 'PI' (Probability of improvement)
#    not_yet_implemented = True
#
#elif acquisition_function == 'PI': # 'UCB' (Upper Confidence Bound), 'EI' (Expected Improvment), 'PI' (Probability of improvement)
#    not_yet_implemented = True



#=======================================================
#         Search best candidate(s) through Heap queue algorithm
#=======================================================

if type_of_optimization == 'maximize':
    top_candidates = heapq.nlargest(keep_how_many_candidates, candidates_acq, key=lambda e:e[0])
else: # 'minimize'
    top_candidates = heapq.nsmallest(keep_how_many_candidates, candidates_acq, key=lambda e:e[0])

candidates_header = [["Acquisition function", "Which row (starts from 1) of materials from unobserved list"]]
top_candidates = candidates_header + top_candidates



#=======================================================
#         Printout
#=======================================================

suggested_candidates_file_name = suggested_candidates_file_name_prefix + str(number_of_observed).zfill(4) + ".csv"

with open(suggested_candidates_file_name, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(top_candidates)

# MAE
def calculate_mae(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Calculate absolute errors
    absolute_errors = [abs(a - b) for a, b in zip(list1, list2)]
    
    # Calculate mean of absolute errors
    mae = sum(absolute_errors) / len(absolute_errors)
    
    return mae

if fake_observation:
    with open(complete_input_features_file_name) as f:
        reader = csv.reader(f)
        complete_input_features_file_csv = [row for row in reader]
    
    with open(complete_target_file_name) as f:
        reader = csv.reader(f)
        complete_target_file_csv = [row for row in reader]
    
    assert len(complete_input_features_file_csv) == len(complete_target_file_csv)
    
    #########
    # 正解データと予測データのMAEの算出

    # 正解データをクロールする
    complete_target_list = []
    pred_target_list = []
    pred_target_std_list = []
    absolute_index_list = []
    complete_dict = {}
    for matID in range(len(complete_input_features_file_csv)):
        complete_dict[tuple(complete_input_features_file_csv[matID])] = matID
    for mat_ID2 in range(number_of_unobserved):
        # position_complete_data = -1
        position_complete_data = complete_dict[tuple(unobserved_input_features_csv[mat_ID2])] # keyがなければエラー
        # for matID in range(len(complete_input_features_file_csv)):
        #     if complete_input_features_file_csv[matID] == unobserved_input_features_csv[mat_ID2]:
        #         position_complete_data = mat_ID
        # assert position_complete_data>=0
        complete_target_list.append(complete_target_file_csv[position_complete_data])
        pred_target_list.append(candidates_avg[mat_ID2])
        pred_target_std_list.append(candidates_std[mat_ID2])
        absolute_index_list.append(position_complete_data)

    complete_target_list = [float(t[0]) for t in complete_target_list]
    # print(complete_target_list)
    # print(pred_target_list)

    with open(f"true_estimate_{str(number_of_observed).zfill(4)}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header (optional)
        writer.writerow(["true", "estimate", "mat_index"])
        
        # Write the data rows
        for item1, item2, item3 in zip(complete_target_list, pred_target_list, absolute_index_list):
            writer.writerow([item1, item2, item3])
    # stdも書き出す
    with open(f"true_estimate_std_{str(number_of_observed).zfill(4)}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header (optional)
        writer.writerow(["true", "estimate", "mat_index"])
        
        # Write the data rows
        for item1, item2, item3 in zip(complete_target_list, pred_target_std_list, absolute_index_list):
            writer.writerow([item1, item2, item3])
    mae = calculate_mae(complete_target_list, pred_target_list)
    print("MAE:", mae)
    mse = calculate_mse(complete_target_list, pred_target_list)
    print("MSE:", mse)

    import csv
    # unobserved_prediction.csvに結果を追加
    unobserved_prediction_file = 'unobserved_prediction.csv'

    # ファイルが存在しない場合はヘッダーを追加
    file_exists = os.path.isfile(unobserved_prediction_file)

    with open(unobserved_prediction_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["MAE", "MSE"])
        writer.writerow([mae, mse])
    
    position_of_candidates_from_unobserved = top_candidates[1][1] # before shuffle
    position_of_candidates_from_complete_data = -1
    count_matched = 0
    maximum_similarity = 0.0
    
    print (complete_input_features_file_csv[0])
    print (unobserved_input_features_csv[position_of_candidates_from_unobserved])
    
    for mat_ID in range(len(complete_input_features_file_csv)):
        if complete_input_features_file_csv[mat_ID] == unobserved_input_features_csv[position_of_candidates_from_unobserved]:
            position_of_candidates_from_complete_data = mat_ID
            count_matched += 1
        
        #similarity_temp = cosine_similarity([complete_input_features_file_csv[mat_ID]], [unobserved_input_features_csv[position_of_candidates_from_unobserved]])
        #if similarity_temp > maximum_similarity:
        #    print (similarity_temp)
        #    maximum_similarity = similarity_temp
        #    position_of_candidates_from_complete_data = mat_ID
    
    assert position_of_candidates_from_complete_data >= 0 # Failed to find matched material
    #assert count_matched == 1 # Find multiple matched material
    assert count_matched >= 1 # Possiblely no problem in some cases... (Redundant data)
    
    
    observed_input_features_csv.append(unobserved_input_features_csv.pop(position_of_candidates_from_unobserved))
    observed_target_csv.append(complete_target_file_csv[position_of_candidates_from_complete_data])
    
    #write observed input
    with open(observed_input_features_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(observed_input_features_csv)
    
    #write unobserved input
    with open(unobserved_input_features_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(unobserved_input_features_csv)
    
    #write observed target
    with open(observed_target_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(observed_target_csv)
    
