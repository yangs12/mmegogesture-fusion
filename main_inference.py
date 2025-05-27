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
from utils.result_utils import *
from model import mainattention_Net, mainlate_Net, main_Net2_quant

@hydra.main(version_base=None, config_path="conf", config_name="config_inference_quant")
def main(inf_args: DictConfig) -> None:
  config = OmegaConf.to_container(inf_args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  path_args = config['path_args']
  path_model = config['path_model']
  args = OmegaConf.load(path_args)

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

  if 'cross' in args.model.fusion:
    model = mainattention_Net.MyNet_Main(args,device)
  elif 'late' in args.model.fusion:
    model = mainlate_Net.MyNet_Main(args,device)
  else:
    model = main_Net2_quant.MyNet_Main(args,device)

  # Load model
  model.load_state_dict(torch.load(path_model))
  model.eval().to(device)

  trainer = Trainer(model=model, 
                  data_train=data_train, 
                  data_valid=[],
                  data_test=data_test,
                  args=args, 
                  device=device,
                  )
  loss_fn=torch.nn.CrossEntropyLoss().to(device)
  test_acc, test_loss, test_y, test_y_pred, test_y_prob, test_des = trainer.test(data_test, device, model, loss_fn, 0)
  print(f"Test Accuracy: {test_acc:.4f}")
  # save_result_confusion(test_y, test_y_pred, trainer.label, 'test-confusion', config['path_save'])

if __name__ == '__main__':
  main()