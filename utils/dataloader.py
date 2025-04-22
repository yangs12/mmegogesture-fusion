import os
import copy
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .transform import *
from .dataset import Dataset_Gesture
from .camera import *
from .result_utils import *
# random.seed(144000)

def Visualize_data(args):       # (TODO)
    # load & clean des
    des_all = pd.read_csv(args.result.path_des, index_col=0)
    val_idx=((des_all['Remark_Snapshot']!='Issue')
            )
    des_clean = des_all[val_idx]
    sensor = [args.sensor.select]

    sub_sel, env_sel, ord_sel = 'Subject2', 'Environment1', 4
    data_dir=args.result.path_data
    for idx in range(len(des_clean)):
        des_snapshot = des_clean.iloc[idx]
        episode = des_snapshot['Episode']
        ord     = des_snapshot['Order']
        sub     = des_snapshot['Subject']
        env     = des_snapshot['Enviroment']
        gesture = des_snapshot['Gesture']
        data = {}
        for sensor_sel in sensor:
            sensor_sel = sensor_sel.split(':')[0]
            # time_s = time.time()
            data_path = os.path.join(data_dir,f'{episode}-{ord}-{sensor_sel}.npy')
            data[sensor_sel] = np.load(data_path)
            show_RGB_image(data['cam'], path=args.result.path_save_vis+'/vis_image/', name=f'{env}-{sub}-{gesture}-{ord}.jpg')
            # if sub_sel==sub and ord_sel==ord and env_sel==env:
            #     # showRGB_video(data['cam'], path=args.result.path_save_vis+'/vis_camera/', name=f'{env}-{sub}-{gesture}-{ord}.mp4')
            #     show_radar(data['rad-uD_channel'], path=args.result.path_save_vis+'/vis_rad-uD/', name=f'{env}-{sub}-{gesture}-{ord}.jpg')
        data['des'] = des_snapshot

def Analyze_statistics(args, transform_list):
    """
    Preprocess & Save the pre-processed data
    """
    # load & clean des
    des_all = pd.read_csv(args.result.path_des, index_col=0)
    val_idx=((des_all['Remark_Snapshot']!='Issue')
            )
    des_clean = des_all[val_idx]
    sensor = [args.sensor.select] if str(type(args.sensor.select))=="<class 'str'>" else args.sensor.select
    
    compose_transform = transforms.Compose(transform_list['test'])

    data_dir=args.result.path_data
    data_all = {}
    for sensor_sel in sensor:
        data_all[sensor_sel] = []
    
    print(f'Calculate Statistics')
    for idx in tqdm(range(len(des_clean))):
        des_snapshot = des_clean.iloc[idx]
        episode = des_snapshot['Episode']
        order   = des_snapshot['Order']
        gesture = des_snapshot['Gesture']
        data = {}
        for sensor_sel in sensor:
            # time_s = time.time()
            sensor_name = sensor_sel
            data[sensor_sel] = sensordata_load((episode, order, sensor_name), data_dir)
            # print(f'{sensor_sel}: {time.time()-time_s}')
        data['des'] = des_snapshot
        data = compose_transform(data)
        for sensor_sel in sensor:
            data_all[sensor_sel].append(data[sensor_sel])
    for sensor_sel in sensor:
        data_all[sensor_sel] = torch.stack(data_all[sensor_sel])
    
    # Apply calculated statistics into actual mean-std list
    for sensor_sel in sensor:
        if 'rad' in sensor_sel:
            args.transforms.mean_std[sensor_sel] = [data_all[sensor_sel].mean().item(), data_all[sensor_sel].std().item()]
        elif 'cam' in sensor_sel:
            args.transforms.mean_std[sensor_sel] = [[data_all[sensor_sel][:,channel,...].mean().item(), data_all[sensor_sel][:,channel,...].std().item()] 
                                                    for channel in range(3)]
        print(f'Updated mean and std of {sensor_sel}')
    return


def LoadDataset_Gesture(args, transform_list):
    """Do transforms on radar data and labels. Load the data from 2 radar sensors.

    Args:
        args: args configured in Hydra YAML file

    """
    # load & clean des
    des_all   = pd.read_csv(args.result.path_des, index_col=0)
    val_idx=((des_all['Remark_Snapshot']!='Issue')
            )
    des_clean = des_all[val_idx]
    sensor = [args.sensor.select] if str(type(args.sensor.select))=="<class 'str'>" else args.sensor.select

    # train-test split
    des_clean = des_clean.to_dict('records')
    ## random split
    des_train, des_test = split_traintest(des_clean, args.train.traintest_split)

    transform_train = transform_list['train'] + [NormalizeTensor(mean_std=args.transforms.mean_std, apply=sensor)]
    transform_test  = transform_list['test'] + [NormalizeTensor(mean_std=args.transforms.mean_std, apply=sensor)]
    ### Compose the transforms on train set
    data_train = Dataset_Gesture(
                                args=args,
                                des=des_train,
                                data_dir=args.result.path_data,
                                flag='train',
                                transform = transforms.Compose(transform_train), 
                                )
    data_test  =  Dataset_Gesture(
                                args=args,  
                                des=des_test, 
                                data_dir=args.result.path_data,
                                flag='test',
                                transform = transforms.Compose(transform_test),
                                )

    data_train = DataLoader(data_train, collate_fn=my_collate_fn, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers)
    data_test = DataLoader(data_test, collate_fn=my_collate_fn, batch_size=int(np.max((args.train.batch_size//4,1))), shuffle=False, num_workers=args.train.num_workers)

    return data_train, data_test

def my_collate_fn(batch):
    collate_data = []
    for idx in range(len(batch[0])):
        collate_data.append([])
    for sample in batch:
        for idx in range(len(sample)):
            collate_data[idx].append(sample[idx])
    for idx in range(len(collate_data)):
        if isinstance(collate_data[idx][0], list) or isinstance(collate_data[idx][0], dict):    # for dict
            collate_data[idx] = collate_data[idx]
        elif isinstance(collate_data[idx][0], int):     # label
            collate_data[idx] = torch.tensor(collate_data[idx])
        else:   # if data is torch
            collate_data[idx] = torch.stack(collate_data[idx])
    return collate_data