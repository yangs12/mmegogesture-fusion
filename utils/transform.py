import numpy as np
import pandas as pd
import copy
import cv2
import math
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
### Radar data Transforms

class ToCHWTensor(object):
    """
    Convert numpy array to CHW tensor format.
    Output format: (C x H x W) or (C x T x H x W)
    """
    def __init__(self, apply):
        self.sensor = apply

    def __call__(self, dat):
        for sensor_sel in self.sensor:
            dat_output  = dat[sensor_sel]
            if isinstance(dat_output, np.ndarray):
                dat_output = torch.from_numpy(dat_output)
            if 'cam' in sensor_sel:
                if not torch.is_tensor(dat_output):
                    dat_output = T.pil_to_tensor(dat_output)
                # dat_output = dat_output.permute(3,0,1,2)/255.
                dat_output = dat_output/255.
            if 'channel' in sensor_sel:
                dat_output = dat_output.permute(2,0,1)
            dat[sensor_sel] = dat_output
        return dat

class RandomizeCrop_Time(object):
    """
    Randomly select starting time
    """
    def __init__(self, win_snapshot, t_snapshot, t_actual, t_spare, apply):
        (self.cam_snapshot, self.rad_snapshot) = win_snapshot
        self.t_snapshot   = t_snapshot
        self.t_actual     = t_actual
        self.t_spare      = t_spare
        self.sensor       = apply
    
    def __call__(self, dat):
        for sensor_sel in self.sensor:
            dat_output  = dat[sensor_sel]
            if 'img' not in sensor_sel:
                win_size = self.cam_snapshot if 'cam' in sensor_sel else self.rad_snapshot
                ratio = win_size/self.t_snapshot
                start_center_idx = (self.t_spare+self.t_actual/2)*ratio
                end_center_idx   = win_size - (self.t_spare+self.t_actual/2)*ratio
                if dat['des']['Remark_Snapshot'] is not np.nan:
                    if dat['des']['Order']==0:
                        if 'cam' in sensor_sel:
                            threshold = self.cam_snapshot-dat[sensor_sel].shape[1]
                        else:
                            if 'channel' in sensor_sel:
                                threshold = torch.where(dat_output.mean((0,1))<300)[0][0].item() 
                            else:
                                threshold = torch.where(dat_output.mean(0)<300)[0][0].item()
                        start_center_idx = np.max((start_center_idx, threshold+round((self.t_actual/2)*ratio)))
                        if 'cam' in sensor_sel:
                            dat_nan = torch.ones((dat_output.shape[0],threshold,dat_output.shape[2],dat_output.shape[3]))*torch.nan
                            dat_output = torch.cat((dat_nan,dat_output),axis=1)
                    if dat['des']['Order']>10:
                        if 'cam' in sensor_sel:
                            threshold = dat[sensor_sel].shape[1]
                        else:
                            if 'channel' in sensor_sel:
                                threshold = torch.where(dat_output.mean((0,1))<300)[0][-1].item()
                            else:
                                threshold = torch.where(dat_output.mean(0)<300)[0][-1].item()
                        end_center_idx = np.min((end_center_idx, threshold-round((self.t_actual/2)*ratio)))
                        if 'cam' in sensor_sel:
                            dat_nan = torch.ones((dat_output.shape[0],self.cam_snapshot-threshold,dat_output.shape[2],dat_output.shape[3]))*torch.nan
                            dat_output = torch.cat((dat_output,dat_nan),axis=1)  
                center_idx  = random.randint(math.ceil(start_center_idx),math.floor(end_center_idx))
                if 'cam' in sensor_sel:
                    dat_output = dat_output[:,center_idx-round((self.t_actual/2)*ratio):center_idx+round((self.t_actual/2)*ratio),:,:]
                elif 'channel' in sensor_sel:
                    dat_output = dat_output[:,:,center_idx-round((self.t_actual/2)*ratio):center_idx+round((self.t_actual/2)*ratio)]
                else:
                    dat_output = dat_output[:,center_idx-round((self.t_actual/2)*ratio):center_idx+round((self.t_actual/2)*ratio)]
                assert dat_output.max().item()<300  
            dat[sensor_sel] = dat_output
        return dat
    
class CenterCrop_Time(object):
    """
    Uniformly select starting time idx of radar&keypoint and crop it to N-sec. size, Divide, and Merge Them
    """
    def __init__(self, win_snapshot, t_snapshot, t_actual, apply):
        (self.cam_snapshot, self.rad_snapshot) = win_snapshot
        self.t_snapshot   = t_snapshot
        self.t_actual     = t_actual
        self.sensor       = apply
    
    def __call__(self, dat):
        for sensor_sel in self.sensor:
            dat_output  = dat[sensor_sel]
            if 'img' not in sensor_sel:
                win_size = self.cam_snapshot if 'cam' in sensor_sel else self.rad_snapshot
                ratio = win_size/self.t_snapshot
                center_idx = win_size//2
                if dat['des']['Remark_Snapshot'] is not np.nan:
                    if dat['des']['Order']==0:
                        if 'cam' in sensor_sel:
                            threshold = self.cam_snapshot-dat[sensor_sel].shape[1]
                        else:
                            if 'channel' in sensor_sel:
                                threshold = torch.where(dat_output.mean((0,1))<300)[0][0].item() 
                            else:
                                threshold = torch.where(dat_output.mean(0)<300)[0][0].item() 
                        center_idx += np.max((round((self.t_actual/2)*ratio)-(center_idx-threshold),0))
                        if 'cam' in sensor_sel:
                            dat_nan = torch.ones((dat_output.shape[0],threshold,dat_output.shape[2],dat_output.shape[3]))*torch.nan
                            dat_output = torch.cat((dat_nan,dat_output),axis=1)
                    if dat['des']['Order']>10:
                        if 'cam' in sensor_sel:
                            threshold = dat[sensor_sel].shape[1]
                        else:
                            if 'channel' in sensor_sel:
                                threshold = torch.where(dat_output.mean((0,1))<300)[0][-1].item()
                            else:
                                threshold = torch.where(dat_output.mean(0)<300)[0][-1].item()  
                        center_idx -= np.max((round((self.t_actual/2)*ratio)+center_idx-threshold,0))
                        if 'cam' in sensor_sel:
                            dat_nan = torch.ones((dat_output.shape[0],self.cam_snapshot-threshold,dat_output.shape[2],dat_output.shape[3]))*torch.nan
                            dat_output = torch.cat((dat_output,dat_nan),axis=1)
                if 'cam' in sensor_sel:
                    dat_output = dat_output[:,center_idx-round((self.t_actual/2)*ratio):center_idx+round((self.t_actual/2)*ratio),:,:]
                elif 'channel' in sensor_sel:
                    dat_output = dat_output[:,:,center_idx-round((self.t_actual/2)*ratio):center_idx+round((self.t_actual/2)*ratio)]
                else:
                    dat_output = dat_output[:,center_idx-round((self.t_actual/2)*ratio):center_idx+round((self.t_actual/2)*ratio)]
                assert dat_output.max().item()<300
            dat[sensor_sel] = dat_output
        return dat

class ResizeTensor(object):
    """
    - Resize data
    """
    def __init__(self, size_rad, size_cam, apply):
        self.size_rad = (size_rad[0],size_rad[1])
        self.size_cam = (size_cam, size_cam)
        self.sensor = apply
    def __call__(self, dat):
        for sensor_sel in self.sensor:
            dat_output = dat[sensor_sel]
            if 'cam' in sensor_sel:
                if 'vid' in sensor_sel:
                    dat_output = dat_output.permute((1,0,2,3))
                    dat_output = F.interpolate(dat_output, self.size_cam )
                    dat_output = dat_output.permute((1,0,2,3))
                elif 'img' in sensor_sel:
                    dat_output = F.interpolate(dat_output.unsqueeze(dim=0), self.size_cam).squeeze(dim=0)
            elif 'rad' in sensor_sel:
                if 'channel' in sensor_sel:
                    dat_output = F.interpolate(dat_output.unsqueeze(dim=0), self.size_rad).squeeze(dim=0)
                else:
                    dat_output = F.interpolate(dat_output.view(1,1,dat_output.shape[0],dat_output.shape[1]), self.size_rad).squeeze()
            dat[sensor_sel] = dat_output
        return dat
        
class ResampleVideo(object):
    """
    Resample Video in time dimension
    """
    def __init__(self, resample, apply):
        self.resample   = resample
        self.sensor     = apply
    def __call__(self, dat):
        for sensor_sel in self.sensor:
            dat_output = dat[sensor_sel]
            if 'vid' in sensor_sel:
                n_max = dat_output.shape[-3] - 1
                indices = torch.linspace(0, n_max, self.resample).long()
                down_rate = int(1/self.resample)
                dat_output = torch.index_select(dat_output, -3, indices)
            dat[sensor_sel] = dat_output
        return dat

class CropDoppler(object):
    """
    Crop Doppler data in Doppler dimension
    """
    def __init__(self, ratio, apply):
        self.ratio   = ratio
        self.sensor     = apply
    def __call__(self, dat):
        for sensor_sel in self.sensor:
            dat_output = dat[sensor_sel]
            if 'uD' in sensor_sel:
                resize_dim   = round(dat_output.shape[0]*self.ratio)
                center_point = dat_output.shape[0]//2
                dat_output = dat_output[center_point-resize_dim//2:center_point+resize_dim//2,:]
            dat[sensor_sel] = dat_output
        return dat

class NormalizeTensor(object):
    """
    Apply z-normalization
    """
    def __init__(self, mean_std, apply):
        self.mean_std   = mean_std
        self.sensor     = apply
    def __call__(self, dat):
        for sensor_sel in self.sensor:
            mean_std = self.mean_std[sensor_sel]
            dat_output = dat[sensor_sel]
            if len(mean_std)>2:
                for c_idx in range(len(mean_std)):
                    dat_output[c_idx] = (dat_output[c_idx]-mean_std[c_idx][0])/mean_std[c_idx][1]
            else:
                dat_output = (dat_output-mean_std[0])/mean_std[1]
            dat[sensor_sel] = dat_output
        return dat