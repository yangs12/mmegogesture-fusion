"""
Datasets for gaits, identity, location, and velocities
"""

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import h5py
import pickle
import joblib
from scipy import signal
from .result_utils import *

def read_h5_basic(path):
    """Read HDF5 files

    Args:
        path (string): a path of a HDF5 file

    Returns:
        radar_dat: micro-Doppler data with shape (256, 128, 2) as (1, time, micro-Doppler, 2 radar channel)
        des: information for radar data
    """
    hf = h5py.File(path, 'r')
    radar_dat = np.array(hf.get('radar_dat'))
    radar_rng = np.array(hf.get('radar_rng'))
    des = dict(hf.attrs)
    hf.close()
    return radar_dat, radar_rng, des

class Dataset_Gesture(Dataset):
    """
    """
    def __init__(self,
                 args,
                 des,
                 data_dir,
                 flag,
                 transform=None,
                 ):
        self.des = des
        self.data_dir = data_dir
        self.flag = flag
        self.transform = transform

        self.sensor = [args.sensor.select] if str(type(args.sensor.select))=="<class 'str'>" else args.sensor.select

    def __len__(self):
        return len(self.des)    

    def __getitem__(self, idx):
        des_snapshot = self.des[idx]
        episode = des_snapshot['Episode']
        order   = des_snapshot['Order']
        gesture = des_snapshot['Gesture']
        data = {}
        for sensor_sel in self.sensor:
            sensor_name = sensor_sel
            data[sensor_sel] = sensordata_load((episode, order, sensor_name), self.data_dir)
        data['des'] = des_snapshot

        if self.transform:
            data = self.transform(data)

        label = int(gesture.split('e')[-1])-1
        del(data['des'])
        output = []
        output = [data[key] for key in data.keys()]
        output.append(label)
        output.append(des_snapshot)
        return output
        