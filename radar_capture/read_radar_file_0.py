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
from omegaconf import OmegaConf
import time


@hydra.main(version_base="1.2", config_path="hyperparam_configs", config_name="read_radar")
# args are imported from the config file - hyperparam_configs/tracking_config.yaml
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False) # allowing adding new keys to args
    # Start timing
    start_time = time.time()
    os.makedirs(args.output_resize_dir, exist_ok=True)
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

    # Some parameters
    uD_axis = np.arange(-args.n_uD_fft//2, args.n_uD_fft//2)*2*radar_params['velocity_max']/args.n_uD_fft  
    args.uD_bins_ps = int(radar_params['n_chirps']/(args.n_uD_fft*(1-args.overlap_ratio)) * radar_params['fps'])  
    print(f"uD bins per second: {args.uD_bins_ps}")

    # Processing each radar file
    for radar_file_path in radar_files:
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

        uD = RD(radar_cube, args, if_stft=True, window=False)
        print("uD shape: ", uD.shape)

        # np.save(os.path.join(args.output_dir, f'{radar_file_name}-0-rad-uD.npy'), uD)
        # save_uD_plot(uD, args, radar_file_name, uD_axis)

        if args.if_resize_uD:
            # Resize the uD to the desired size
            uD_resized = cv2.resize(uD, (args.uD_width, args.uD_height))
            print("Resized uD shape: ", uD_resized.shape)
            # os.makedirs(os.path.join(args.output_dir, 'uD_resized'), exist_ok=True)
            np.save(os.path.join(args.output_resize_dir, f'{radar_file_name}-0-rad-uD.npy'), uD_resized)
            
            save_uD_plot(uD_resized, args, radar_file_name+'_resized', uD_axis)
        if args.if_resize_img:
            vide_path = os.path.join(args.folder_path, 'camera', radar_file_name + ".avi")
            print(f"Video path: {vide_path}")
            # Load the video
            cap = cv2.VideoCapture(vide_path)
            if not cap.isOpened():
                print("Error opening video file")
                continue
            # Read the first frame
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            print(f"Total frames loaded: {len(frames)}")
            # if not ret:
            #     print("Error reading video frame")
            #     continue
            # Calculate the middle frame index for a 2.667s video
            middle_frame_index = int(args.capture_time * args.cam_fps / 2)

            # Ensure the index is within bounds
            if middle_frame_index >= len(frames):
                print("Middle frame index exceeds total frames, skipping.")
                continue

            # Extract the middle frame
            img = frames[middle_frame_index]
            # Resize the image to the desired size
            img_resized = cv2.resize(img, (args.img_width, args.img_height))
            print("Resized image shape: ", img_resized.shape)
            # os.makedirs(os.path.join(args.output_dir, 'img_resized'), exist_ok=True)
            np.save(os.path.join(args.output_resize_dir, f'{radar_file_name}-0-img.npy'), img_resized)
            # Save the resized image
            # save_img_path = os.path.join(args.output_dir, 'img_resized', f'{radar_file_name}.png')
            # os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir, f'{radar_file_name}-0-img.png'), img_resized)
            # print(f"Resized image saved at: {save_img_path}")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()