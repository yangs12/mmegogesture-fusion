'''
This file tracks people using the radar data and saves the tracking information in a json files
'''
from utils.xwr_raw.radar_config import RadarConfig
from utils.pipeline_utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import hydra
from omegaconf.dictconfig import DictConfig
import copy
import cv2
# import shutil
import pandas as pd
import json
import time



@hydra.main(version_base="1.2", config_path="hyperparam_configs", config_name="read_radar")
# args are imported from the config file - hyperparam_configs/tracking_config.yaml
def main(args: DictConfig) -> None:
    file_date = args.radar_data_path.split("/")[-3].split(".")[0]+"/"
    print(f"Processing {file_date} folder")

    # If one file is given, process that file only, otherwise process all files in the folder
    if args.radar_file_name:
        radar_file_path = args.radar_data_path + str(args.radar_file_name) + ".npz"
        radar_files = [args.radar_data_path + str(args.radar_file_name) + ".npz"]
    else:
        radar_files = [os.path.join(args.radar_data_path, f) for f in os.listdir(args.radar_data_path) if f.endswith('.npz')]

        
    # Read radar configurations and radar parameters
    with open(args.radar_config_path, 'r') as f:
        cfg = f.readlines()
    radar_config = RadarConfig(cfg)
    radar_params = radar_config.get_params()

    # Processing each radar file
    for radar_file_path in radar_files:
        start_time = time.time()
        radar_file_name = (radar_file_path.split("/")[-1]).replace(".npz", "")
        print(f"\n**Processing {radar_file_path}**")

        # --- Load the radar data ---
        '''
        Return: radar_cube - raw ADC data (num_frames, num_chirps, num_range_bins, num_doppler_bins), 
                pcloud_list - list of point clouds
                info - some header info
                r_axis - range axis based on radar params, e.g., range resolution and max range
                d_axis - doppler axis, based on doppler resolution and max doppler
        '''
        radar_loader = radarDataLoader(radar_file_path, radar_params)
        radar_cube, pcloud_list, info, num_frames, r_axis, d_axis = radar_loader.load_data()
        
        # keep the capture period frames - since we capture slightly more than the data needed
        valid_radar_frames = radar_cube.shape[0]
        print(f"Number of frames in the radar cube: {valid_radar_frames}")
        radar_cube = radar_cube[:int(args.capture_time * radar_params['fps']+1),...]
        num_radar_frames = radar_cube.shape[0]
        print("Final radar_cube shape: ", radar_cube.shape)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    main()