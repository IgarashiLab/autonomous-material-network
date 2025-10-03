import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

import sys

parser = argparse.ArgumentParser(description="Script to split the complete dataset into observed and unobserved data")

parser.add_argument("--observed_num", "-o", type=int, help="Number of data points to be observed. Default=10")
parser.add_argument("--random","-r", action="store_true", help="Whether to split the data randomly")
parser.add_argument("--seed", "-s", type=int, help="Random seed value for splitting. Default=0")

args = parser.parse_args()

observed_num=10
random_flag=False
random_seed=0


if args.observed_num is not None:
    observed_num=args.observed_num
if args.random:
    random_flag=True
if args.seed is not None:
    random_seed=args.seed


# Load the complete dataset
input_features_df = pd.read_csv('complete_input_features.csv', header=None)
target_df = pd.read_csv('complete_B2_target.csv', header=None)

# Split the data into training and testing sets
input_features_train, input_features_test, target_train, target_test = train_test_split(input_features_df, target_df, train_size=observed_num, random_state=random_seed, shuffle=random_flag)
# if random_seed is not None:
#     # Split randomly
#     input_features_train, input_features_test, target_train, target_test = train_test_split(input_features_df, target_df, train_size=observed_num, random_state=random_seed, shuffle=True)
# else:
#     # Split sequentially from the top
#     input_features_train = input_features_df.iloc[:observed_num, :]
#     target_train = target_df.iloc[:observed_num, :]
#     input_features_test = input_features_df.iloc[observed_num:, :]
#     target_test = target_df.iloc[observed_num:, :]


# print(input_features_train)

# Save the split data
input_features_train.to_csv('observed_input_features.csv', index=False, header=None)
target_train.to_csv('observed_target.csv', index=False, header=None)
input_features_test.to_csv('unobserved_input_features.csv', index=False, header=None)
