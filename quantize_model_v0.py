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

@hydra.main(version_base=None, config_path="conf", config_name="config_quant")
def main(args: DictConfig) -> None:
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cpu')

  model = main_Net.MyNet_Main(args, device)
  model.load_state_dict(torch.load(args.model_path))
  model.eval()
  # for name, param in model.named_parameters():
  #   print(f"{name}: {param.dtype}")  # all float32


  model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  # qint8 for weights and quint8 for activations
  # FBGEMM backend, which is an optimized int8 kernel for x86 CPUs
  qconfig = torch.quantization.get_default_qconfig('fbgemm')
  print('------------')
  print("Activation observer:", qconfig.activation)
  print("Weight observer:", qconfig.weight)
  print("Activation dtype:", qconfig.activation().dtype)
  print("Weight dtype:", qconfig.weight().dtype)


  torch.quantization.prepare(model, inplace=True)
  torch.quantization.convert(model, inplace=True)
  torch.save(model, args.result_dir+'model_quantized_v0.pt')


if __name__ == '__main__':
  main()
