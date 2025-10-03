import argparse
import os
import subprocess
import re
import numpy as np
import shutil
import heapq
from ruamel.yaml import YAML
import copy


def main():
    yaml=YAML()
    yaml.preserve_quotes=True
    with open("setting.yaml") as f:
        setting_yaml=yaml.load(f)

    comp_input_feature_name = setting_yaml["comp_input_feature_name"]

    system_list=["CT","MM","SP"]
    setting_dict={}
    for name in system_list:
        setting_dict[name]=copy.deepcopy(setting_yaml["common"])
        # Overwrite if there are individual settings
        if setting_yaml[name] is not None:
            for key in setting_yaml["common"].keys():
                if (key in setting_yaml[name]) and (setting_yaml[name][key] is not None):
                    setting_dict[name][key]=setting_yaml[name][key]
            if "add_seed" in setting_yaml[name]:
                setting_dict[name]["seed"]+=setting_yaml[name]["add_seed"]

    print("Settings")
    print(setting_dict)


    # Initialize all systems
    subprocess.run(["python3", "initialize_BO_system.py", "-s", "BO_system_CT", "-o", "CT", "-n", str(setting_dict["CT"]["observed_num"]), "-r", str(setting_dict["CT"]["seed"]), "-f", comp_input_feature_name])
    subprocess.run(["python3", "initialize_BO_system.py", "-s", "BO_system_MM", "-o", "MM", "-n", str(setting_dict["MM"]["observed_num"]), "-r", str(setting_dict["MM"]["seed"]), "-f", comp_input_feature_name])
    subprocess.run(["python3", "initialize_BO_system.py", "-s", "BO_system_SP", "-o", "SP", "-n", str(setting_dict["SP"]["observed_num"]), "-r", str(setting_dict["SP"]["seed"]), "-f", comp_input_feature_name])

    time_line=[]
    # max_time=100 # Maximum time for the timeline
    max_time=setting_yaml["max_time"] # Maximum time for the timeline
    # Definition of tuple. (Start time, required time (setting), system_path, count until the next transfer_mode, initial count value)
    # heapq.heappush(time_line, (0, 1, "BO_system_CT", -1, -1))
    # heapq.heappush(time_line, (0, 1, "BO_system_MM", -1, -1))
    # heapq.heappush(time_line, (0, 1, "BO_system_SP", -1, -1))
    # heapq.heappush(time_line, (0, 1, "BO_system_CT", 5, 5))
    # heapq.heappush(time_line, (0, 1, "BO_system_MM", 5, 5))
    # heapq.heappush(time_line, (0, 1, "BO_system_SP", 5, 5))
    for name in system_list:
        heapq.heappush(time_line, (
            setting_dict[name]["process_time"]["head"],
            setting_dict[name]["process_time"]["reset"],
            f"BO_system_{name}", 
            setting_dict[name]["transfer_interval"]["head"], 
            setting_dict[name]["transfer_interval"]["reset"]))

    while time_line:
        process_tuple = heapq.heappop(time_line)
        if process_tuple[0]>=max_time:
            continue
        system_path=process_tuple[2]
        mode="no_transfer" if process_tuple[3]!=0 else "transfer"
        print(process_tuple[0], system_path, mode)
        subprocess.run(["python3", "run_oneBO.py", "-s", system_path, "-m", mode])

        count = process_tuple[3]-1 if process_tuple[3]>0 else process_tuple[4]
        heapq.heappush(time_line, (process_tuple[0]+process_tuple[1], process_tuple[1], process_tuple[2], count, process_tuple[4]))

if __name__ == '__main__':
    main()