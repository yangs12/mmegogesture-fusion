"""
Main File
"""
import os
import sys
import hydra
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from utils.trainer import Trainer
from utils.dataloader import *
from model import mobileVit, main_Net

@hydra.main(version_base=None, config_path="conf", config_name="config_gesture")
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
  # Preprocessing & Save
  if args.preprocess.flag_statistics:
    Analyze_statistics(args, transform_list)
  if args.preprocess.flag_visualize:
    Visualize_data(args)

  data_train, data_test = LoadDataset_Gesture(args, transform_list)

  if args.wandb.use_wandb:
    wandb.init(
          project = args.wandb.project, 
          entity = "shuboy", 
          config = config, 
          notes = args.result.note,
          name = args.result.name
          )
    wandb.config = {
          "learning_rate": args.train.learning_rate,
          "weight_decay": args.train.weight_decay,
          "delta": args.train.delta,
          }
  
  if 'mobileVit' in args.model.backbone:
    model = mobileVit.main_Net(args).to(device)
  else:
    model = main_Net.MyNet_Main(args,device)
  # Learning
  trainer = Trainer(model=model, 
                    data_train=data_train, 
                    data_valid=[],
                    data_test=data_test,
                    args=args, 
                    device=device,
                    )
  trainer.train()

  wandb.finish()


if __name__ == '__main__':
  main()

# from clone.calculateflops.calflops import flops_counter
# batch_size = 1
# input_shape = (batch_size, 3, 16, 256, 256)
# flops, macs, params = flops_counter.calculate_flops(model=model, 
#                                     input_shape=input_shape,
#                                     output_as_string=True,
#                                     output_precision=4)
