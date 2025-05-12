import os
import sys
import hydra
import wandb
import tqdm
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from utils.trainer import Trainer
from utils.dataloader import *
from model import mobileVit, main_Net, mainattention_Net

@hydra.main(version_base=None, config_path="conf", config_name="config_gesture_attention")
def main(args: DictConfig) -> None:
    config = OmegaConf.to_container(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sensor = [args.sensor.select] if str(type(args.sensor.select))=="<class 'str'>" else args.sensor.select
    transform_list = {'train':[
                        ToCHWTensor(apply=sensor),
                        RandomizeCrop_Time(win_snapshot=(args.sensor.win_cam,args.sensor.win_rad), 
                                            t_snapshot=args.sensor.t_snapshot,
                                            t_actual=args.transforms.t_actual, 
                                            t_spare=args.transforms.t_spare, 
                                            apply=sensor),
                        ResampleVideo(resample=args.transforms.resample_cam, apply=sensor),     # RGB only
                        CropDoppler(ratio=args.transforms.cropratio_uD, apply=sensor),          # radar-uD only
                        ResizeTensor(size_rad=(args.transforms.size_rad_D,args.transforms.size_rad_T), size_cam=args.transforms.size_cam, apply=sensor),
                        ],
                        'test': [
                                ToCHWTensor(apply=sensor),
                                CenterCrop_Time(win_snapshot=(args.sensor.win_cam,args.sensor.win_rad),
                                                t_snapshot=args.sensor.t_snapshot, 
                                                t_actual=args.transforms.t_actual, 
                                                apply=sensor),
                                ResampleVideo(resample=args.transforms.resample_cam, apply=sensor),     # RGB only
                                CropDoppler(ratio=args.transforms.cropratio_uD, apply=sensor),          # radar-uD only
                                ResizeTensor(size_rad=(args.transforms.size_rad_D,args.transforms.size_rad_T), size_cam=args.transforms.size_cam, apply=sensor),
                            ]}

    data_train, data_test = LoadDataset_Gesture(args, transform_list)
    progress_bar = tqdm(data_train)

    sensor = [args.sensor.select] if str(type(args.sensor.select))=="<class 'str'>" else args.sensor.select

    for iter, data in enumerate(progress_bar):
        x_batch = {}
        for sensor_idx, sensor_sel in enumerate(sensor):
            x_batch[sensor_sel] = data[sensor_idx]
        y_batch   = data[-2]
        des_batch = data[-1]
        # print(x_batch)
        print(sensor[0])
        print(x_batch[sensor[0]].shape)
        print(sensor[1])
        print(x_batch[sensor[1]].shape)
        print(y_batch.shape)
        if iter == 0:
            break    

if __name__ == '__main__':
  main()

