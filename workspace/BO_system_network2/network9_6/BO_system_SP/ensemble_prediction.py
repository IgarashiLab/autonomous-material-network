import configparser
import csv
import random
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout


def denormalization(target, deviation, mean):
    
    return_target = []
    for num_mat in range(len(target)):
        #target[num_mat] = (target[num_mat] * deviation) + mean # Undo normalization. Now 'target' has original values.
        return_target.append( (target[num_mat] * deviation) + mean ) # Undo normalization. Now 'target' has original values.
    
    return return_target



#=======================================================
#         Read configuration file
#=======================================================
# Determine the configuration file name from the arguments
import sys

if len(sys.argv) < 2:
    print("Error: Please provide the configuration file name as an argument.")
    sys.exit(1)

config_filename = sys.argv[1]
print(f"ini file：{config_filename}")

config_ini = configparser.ConfigParser()
config_ini.read(config_filename, encoding='utf-8')
config_ini = configparser.ConfigParser()
config_ini.read(config_filename, encoding='utf-8')



#=======================================================
#         Hyperparameters
#=======================================================

input_features_file_name = str(config_ini.get('RUN', 'input_features_file_name')) # Default="all_data_B2_rem_hwang_20220520_simple_non-feff+zero_order.csv"
input_features_have_header = eval(config_ini.get('RUN', 'input_features_have_header')) # Default=False

coeff_norm_D = np.array(eval(config_ini.get('RUN', 'output_deviation'))) # Default=1.0
coeff_norm_M = np.array(eval(config_ini.get('RUN', 'output_mean'))) # Default=0.0
# Normalize output from predefined deviation and mean, only activate when output_normalization=False

model_load_name_prefix = str(config_ini.get('RUN', 'model_load_name_prefix')) # Default="transferred_model_" # Recommended to end with an underscore(_)

number_of_ensemble = int(config_ini.get('RUN', 'number_of_ensemble')) # Default=10
random_initial = int(config_ini.get('RUN', 'random_initial')) # Default=0
# For example of 20; average and deviation from 20 NNs will be used in Bayesian optimization. Higher is better, but 10 is enough.
length_of_random_seed_text = int(config_ini.get('RUN', 'length_of_random_seed_text')) # Default=3
csv_save_name_prefix = str(config_ini.get('RUN', 'csv_save_name_prefix')) # Default="ensemble_predicted_"
# Recommended to end with an underscore(_)

memory_limit_gb = float(config_ini.get('RUN', 'memory_limit_gb')) # Default=2.0
# Maximum VRAM allocation (unit of GB), ignore if you use CPU



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

np.set_printoptions(threshold=np.inf) # setting for maintenance
#np.set_printoptions(threshold=np.100) # original setting

print ("")
print ("Program started")
print ("")
print ("TIME CHECK")
print (datetime.datetime.now())
print ("")



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
#         Check test set
#         (No training in this script.)
#=======================================================

X_test = input_features[:]


print ("==================================================")
print (" You have ",len(X_test) ,"test data.")
print ("==================================================")
print ("")



#=======================================================
#         Ensemble loop... All of the training can be parallelized
#=======================================================

for ensemble_ID in range(random_initial, random_initial + number_of_ensemble): # Loop for number_of_ensemble
    
    #=======================================================
    #         Load neural network architecture
    #=======================================================
    
    np.random.seed(ensemble_ID)
    
    model = load_model(model_load_name_prefix + str(ensemble_ID).zfill(length_of_random_seed_text) + ".h5")
    
    model.summary()
    model.compile()
    print ("Please make sure that this model has the structure you have expected.")
    print ("")
    
    
    
    #=======================================================
    #         Test and evaluation
    #=======================================================
    
    y_test_predicted = model.predict(X_test)
    y_test_denorm = denormalization(y_test_predicted, coeff_norm_D, coeff_norm_M) # denormalization
    
    #if save_csv :
    save_results = []
    for num_mat in range(len(y_test_predicted)):
        #save_results.append((y_test_denorm[num_mat], y_test_predicted[num_mat]))
        save_results.append((y_test_denorm[num_mat]))
        
    with open(csv_save_name_prefix + str(ensemble_ID).zfill(length_of_random_seed_text) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(save_results)
        
    
    print ("")
    print ("TIME CHECK")
    print (datetime.datetime.now())
    print ("")
    print ("Complete prediction from one NN")
