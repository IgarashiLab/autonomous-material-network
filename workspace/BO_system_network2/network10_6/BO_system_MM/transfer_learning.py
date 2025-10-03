import configparser
import csv
import random
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle


def pencil_shuffle(array, seed):
    # Fisher-Yates shuffle
    random.seed(seed)
    array = list(array)
    empty_array = []
    total_length = len(array)
    for i in range(total_length):
        j = random.randrange(len(array))
        empty_array.append(array.pop(j))
    return np.array(empty_array)


def auto_normalization(target):
    
    deviation = np.std(target, axis=0)
    mean = np.mean(target, axis=0)
    
    for num_mat in range(len(target)):
        target[num_mat] = (target[num_mat] - mean) / (deviation*1.0) # Normalized to have a mean of 0 and a variance of 1.
    
    return target, deviation, mean


def manual_normalization(target, deviation, mean):
    
    for num_mat in range(len(target)):
        target[num_mat] = (target[num_mat] - mean) / (deviation*1.0) # Normalized to have a mean of 0 and a variance of 1.
    
    return target


def denormalization(target, deviation, mean):
    
    return_target = []
    for num_mat in range(len(target)):
        #target[num_mat] = (target[num_mat] * deviation) + mean # Undo normalization. Now 'target' has original values.
        return_target.append( (target[num_mat] * deviation) + mean ) # Undo normalization. Now 'target' has original values.
    
    return return_target



#=======================================================
#         Read configuration file
#=======================================================
# Determine the configuration file name to read from the argument
import sys

if len(sys.argv) < 2:
    print("Error: Please enter a configuration file name as an argument.")
    sys.exit(1)

config_filename = sys.argv[1]
print(f"ini file：{config_filename}")

config_ini = configparser.ConfigParser()
config_ini.read(config_filename, encoding='utf-8')



#=======================================================
#         Hyperparameters
#=======================================================

training_target_file_name = str(config_ini.get('RUN', 'training_target_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
input_features_file_name = str(config_ini.get('RUN', 'input_features_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
input_features_have_header = eval(config_ini.get('RUN', 'input_features_have_header')) # Default=False

random_initial = int(config_ini.get('RUN', 'random_initial')) # Default=0 
# random seed for neural network initializing
mix_random_seed = int(config_ini.get('RUN', 'mix_random_seed')) # Default=0
# random seed for data random shuffle
# You may fix training/validation split with different initial NNs.
length_of_random_seed_text = int(config_ini.get('RUN', 'length_of_random_seed_text')) # Default=3
do_not_shuffle = eval(config_ini.get('RUN', 'do_not_shuffle')) # Default=False
# If True, data will not be shuffled and last n_test data will be used as test set. (Use True when you want to control training/testing split manually)
target_list = eval(config_ini.get('RUN', 'target_list')) # Example=[6, 6]
# target_list = [0, 2, 3, 4, 5, 6] # mag, spin, gap, d, Lattice, Curie
# If pretrained model has multiple outputs but you want to predict single target, then just use same target multiple times like [6, 6]
# NOTE that length of target_list in transfer_learning.py is must be equal to that of pretraining (train_from_scratch.py).

output_normalization = eval(config_ini.get('RUN', 'output_normalization')) # Default=True
# Normalize output features. Highly recommended especially for transfer learning.

output_manual_normalization = eval(config_ini.get('RUN', 'output_manual_normalization')) # Default=False
coeff_norm_D = np.array(eval(config_ini.get('RUN', 'output_deviation'))) # Default=1.0
coeff_norm_M = np.array(eval(config_ini.get('RUN', 'output_mean'))) # Default=0.0
# Normalize output from predefined deiviation and mean, only activate when output_normalization=False

model_load_name_prefix = str(config_ini.get('RUN', 'model_load_name_prefix')) # Default="pretrained_model_" # Recommended to end with an underscore(_)

how_many_frozen_layers = int(config_ini.get('RUN', 'how_many_frozen_layers')) # Default=2
# Determine how many first N hidden layers set to be trainable=False

test_set_ratio = float(config_ini.get('RUN', 'test_set_ratio')) # Default=0.2
# Test set ratio  0.1=10%, 0.2=20%, ... Somewhere between 0.0 and 1.0
assert 0.0 <= test_set_ratio < 1.0
n_test = int(config_ini.get('RUN', 'n_test')) # Default=0
# Manually set test set size if test_set_ratio = 0.0. No test set if test_set_ratio = 0.0 and n_test = 0
epochs = int(config_ini.get('RUN', 'epochs')) # Defalut=2000
# Maximum training epochs
patience_epochs = int(config_ini.get('RUN', 'patience_epochs')) # Deafult=100
# How long to keep training without improving of validation error
batch_size = int(config_ini.get('RUN', 'batch_size')) # Default=8
# Smaller for better accuracy, larger for faster training.
learning_rate = float(config_ini.get('RUN', 'learning_rate')) # Default=0.0005
validation_split = float(config_ini.get('RUN', 'validation_split')) # Default=0.2
# Validation set determines which training epoch is optimal. Somewhere between 0.0 and 1.0
assert 0.0 <= validation_split < 1.0

number_of_ensemble = int(config_ini.get('RUN', 'number_of_ensemble')) # Default=10
# For example of 20; average and deviation from 20 NNs will be used in Bayesian optimization. Higher is better, but 10 is enough.
save_model = eval(config_ini.get('RUN', 'save_model')) # Default=False
# save model file (.h5)
model_save_name_prefix = str(config_ini.get('RUN', 'model_save_name_prefix')) # Default="model_"
# Recommended to end with an underscore(_)
save_csv = eval(config_ini.get('RUN', 'save_csv')) # Default=True
# save predicted results of test set (.csv)
csv_save_name_prefix = str(config_ini.get('RUN', 'csv_save_name_prefix')) # Default="transferred_predicted_"
# Recommended to end with an underscore(_)
if save_model == save_csv == False:
    assert False, (
        "Please save training results at least one format.")

memory_limit_gb = float(config_ini.get('RUN', 'memory_limit_gb')) # Default=2.0
# Maximum VRAM allocation (unit of GB), ignore if you use CPU


"""
input_file_name = "all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
observed_file_name_prefix = "observed_" # If use "observed_", full file name will be "observed_000.csv", "observed_001.csv", ...
"""




#=======================================================
#         GPU Start (limit vram usage by memory_limit_gb)
#=======================================================

try:    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],                        # gpus[0], gpus[1], gpus[2], gpus[3] if use 4 GPUs
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_gb*1024)]) # Logical GPU 1, memory unit = MB
            #[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048), # Logical GPU 1
            #tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]) # Logical GPU 2
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
except Exception:
    print ("GPU not detected")
    pass

#parallelization of training of single NN has no merit ...
#strategy = tf.distribute.get_strategy()



#=======================================================
#         Setup
#=======================================================


number_of_targets = len(target_list)

np.set_printoptions(threshold=np.inf) # numpy setting for maintenance
#np.set_printoptions(threshold=np.100) # original setting

print ("")
print ("Program of transfer learning started")
print ("")
print ("TIME CHECK")
print (datetime.datetime.now())
print ("")



#=======================================================
#         Read training target database
#=======================================================

# glass_data_raw = np.loadtxt('glass_rawdata_'+ str(target_viscosity) +'_cleaned.csv', delimiter=',', dtype='float32')
# This approach is valid only if the file doesn't contain non-numeric values.

with open(training_target_file_name) as f:
    reader = csv.reader(f)
    result_raw_csv = [row for row in reader]


number_of_materials = len(result_raw_csv)

training_target = np.zeros((number_of_materials, number_of_targets), dtype='float32')

for num_mat in range(number_of_materials):
    for num_target in range(number_of_targets):
        training_target[num_mat][num_target] = float(result_raw_csv[num_mat][target_list[num_target]])



#=======================================================
#         Normalize training targets
#=======================================================

if output_normalization:
    training_target, coeff_norm_D, coeff_norm_M = auto_normalization(training_target)
    
    print ('normalization coefficient _D, _M')
    print (coeff_norm_D, coeff_norm_M)

elif output_manual_normalization:
    training_target = manual_normalization(training_target, coeff_norm_D, coeff_norm_M)
    
    print ('manual normalization coefficient _D, _M')
    print (coeff_norm_D, coeff_norm_M)



#=======================================================
#         Read input features
#=======================================================

if input_features_have_header:
    input_features = np.loadtxt(input_features_file_name, delimiter=',', skiprows=1, dtype='float32') # Skip first row (header)
else:
    input_features = np.loadtxt(input_features_file_name, delimiter=',', dtype='float32')
    # This approach is valid only if the file doesn't contain non-numeric values.


length_of_input_features = len(input_features[0])

print ("Length of input features :", length_of_input_features)



#=======================================================
#         Data shuffle with same order
#=======================================================

assert len(training_target) == len(input_features)
if not do_not_shuffle:
    training_target = pencil_shuffle(training_target, mix_random_seed)
    input_features = pencil_shuffle(input_features, mix_random_seed)



#=======================================================
#         Split training/test sets (if you use validation, training set = training data + validation data)
#         (No test set in BO, but it will used for separate already-observed data.)
#=======================================================

if test_set_ratio > 0.0:
    n_test = int(number_of_materials*test_set_ratio)

X_train, X_test = input_features[n_test:], input_features[:n_test]
y_train, y_test = training_target[n_test:], training_target[:n_test]

print ("==================================================")
print (" You have ",len(X_train) ,"training data.")
print (" You have ",len(X_test) ,"test data.")
print ("==================================================")
print ("")



#=======================================================
#         Ensemble loop... All of the training can be parallelized
#=======================================================

for ensemble_ID in range(number_of_ensemble): # Loop for number_of_ensemble
    
    #=======================================================
    #         Load neural network architecture
    #=======================================================
    
    np.random.seed(ensemble_ID)
    
    model = load_model(model_load_name_prefix + str(ensemble_ID).zfill(length_of_random_seed_text) + ".h5")
    
    for frozen_layer in range(how_many_frozen_layers):
        model.layers[frozen_layer].trainable = False
    #model.layers[0].trainable = False # Freeze input - 1st hidden layer
    #model.layers[1].trainable = False # Freeze 1st hidden layer - 2nd hidden layer
    
    model.summary()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    print ("Please make sure that this model has the structure you have expected.")
    print ("")
    
    
    
    #=======================================================
    #         Training a neural network
    #=======================================================
    
    callback = EarlyStopping(monitor='val_loss', patience=patience_epochs, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=validation_split, callbacks=[callback])
    
    
    
    #=======================================================
    #         Test and evaluation
    #=======================================================
    
    y_train_predicted = model.predict(X_train)
    y_train_predicted = denormalization(y_train_predicted, coeff_norm_D, coeff_norm_M) # denormalization
    y_train_denorm = denormalization(y_train, coeff_norm_D, coeff_norm_M) # denormalization
    
    if n_test > 0:
        y_test_predicted = model.predict(X_test)
        y_test_predicted = denormalization(y_test_predicted, coeff_norm_D, coeff_norm_M) # denormalization
        y_test_denorm = denormalization(y_test, coeff_norm_D, coeff_norm_M) # denormalization
        
        if save_csv :
            save_csv_name = csv_save_name_prefix + str(ensemble_ID).zfill(length_of_random_seed_text) + "_test.csv"
            save_results = []
            for num_mat in range(len(y_test_predicted)):
                save_results.append((y_test_denorm[num_mat], y_test_predicted[num_mat]))
            
            with open(save_csv_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(save_results)
        
    if save_csv :
        save_csv_name = csv_save_name_prefix + str(ensemble_ID).zfill(length_of_random_seed_text) + "_train.csv"
        save_results = []
        for num_mat in range(len(y_train_predicted)):
            save_results.append((y_train_denorm[num_mat], y_train_predicted[num_mat]))
        
        with open(save_csv_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(save_results)
    
    
    r2_train = r2_score(y_train_denorm, y_train_predicted, multioutput='raw_values')
    rmse_train = np.sqrt(mean_squared_error(y_train_denorm, y_train_predicted, multioutput='raw_values'))
    mae_train = mean_absolute_error(y_train_denorm, y_train_predicted, multioutput='raw_values')
    
    if n_test > 0:
        r2_test = r2_score(y_test_denorm, y_test_predicted, multioutput='raw_values')
        rmse_test = np.sqrt(mean_squared_error(y_test_denorm, y_test_predicted, multioutput='raw_values'))
        mae_test = mean_absolute_error(y_test_denorm, y_test_predicted, multioutput='raw_values')
    
    if save_model :
        save_model_name = model_save_name_prefix + str(ensemble_ID).zfill(length_of_random_seed_text) + ".h5"
        model.save(save_model_name)
    
    
    print ("==================================================")
    print ("Training R2 is    ", r2_train)
    print ("Training RMSE is  ", rmse_train)
    print ("Training MAE is   ", mae_train)
    if n_test > 0:
        print ("")
        print ("Testing  R2 is     ", r2_test)
        print ("Testing  RMSE is   ", rmse_test)
        print ("Testing  MAE is    ", mae_test)
    print ("==================================================")
    print ("")
    if save_csv:
        print ("csv saved :" + save_csv_name)
    if save_model:
        print ("model saved :" + save_model_name)
    print ("")
    print ("TIME CHECK")
    print (datetime.datetime.now())
    print ("")
    print ("Complete training and testing of one NN")
    print ("")
    print ("")
