"""
Main File, runs inference based on yaml and saves results to csv (for confusion matrix)
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
from utils.trainer_quant import Trainer
from utils.dataloader import *
from utils.result_utils import *

import pandas as pd

from model.classifier_head import FusionClassifierOptions
from torch.ao.nn.quantized import Linear as nnq_Linear
from torch import nn
from torchvision.models.quantization import mobilenet_v2

def load_models(fusion_type, quantized,device,num_classes,quant_type='qnnpack'):
  model_list = []
  if 'concat' in fusion_type:
    model1 = mobilenet_v2(pretrained=True, quantize=quantized)
    model1.classifier = nn.Identity()
    model1.to(device)

    model2 = mobilenet_v2(pretrained=True, quantize=quantized)
    model2.classifier = nn.Identity()
    model2.to(device)

    fusion_classifier = FusionClassifierOptions(1280 * 2, num_classes,dropout=True, batchnorm=True).to(device)
    if quantized:
      fusion_classifier.eval()
      torch.quantization.fuse_modules(fusion_classifier.fc, [
      ['0', '1'],  # Linear + BatchNorm
      ['4', '5'],  # Linear + BatchNorm
      ], inplace=True)
      fusion_classifier.qconfig = torch.quantization.get_default_qconfig(quant_type)
      torch.quantization.prepare(fusion_classifier, inplace=True)
      torch.quantization.convert(fusion_classifier, inplace=True)
      fusion_classifier.to(device)

    model_list = [model1, model2, fusion_classifier]
  elif 'camonly' in fusion_type:
    model1 = mobilenet_v2(pretrained=True, quantize=quantized)
    model1.classifier = nn.Identity()
    model1.to(device)
    
    fusion_classifier = FusionClassifierOptions(1280, num_classes,dropout=True, batchnorm=True).to(device)
    if quantized:
      fusion_classifier.eval()
      torch.quantization.fuse_modules(fusion_classifier.fc, [
      ['0', '1'],  # Linear + BatchNorm
      ['4', '5'],  # Linear + BatchNorm
      ], inplace=True)
      fusion_classifier.qconfig = torch.quantization.get_default_qconfig(quant_type)
      torch.quantization.prepare(fusion_classifier, inplace=True)
      torch.quantization.convert(fusion_classifier, inplace=True)
      fusion_classifier.to(device)
      
    model_list = [model1, fusion_classifier]
  elif 'late' in fusion_type:
    model1 = mobilenet_v2(pretrained=True, quantize=quantized)
    model2 = mobilenet_v2(pretrained=True, quantize=quantized)
    if quantized:
      model1.classifier[1] = nnq_Linear(model1.last_channel, num_classes)
      model2.classifier[1] = nnq_Linear(model2.last_channel, num_classes)
    else:
      model1.classifier[1] = nn.Linear(model1.last_channel, num_classes)
      model2.classifier[1] = nn.Linear(model2.last_channel, num_classes)

    model1.to(device)
    model2.to(device)

    model_list = [model1, model2]
  
  return model_list



@hydra.main(version_base=None, config_path="conf/avgresult_config/late", config_name="late_inference_0129")
def main(inf_args: DictConfig) -> None:
  config = OmegaConf.to_container(inf_args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  path_args = config['path_args']
  path_model = config['path_model']
  quantized = config['quantized']
  args = OmegaConf.load(path_args)

  if quantized:
    quant_type = 'qnnpack'
    torch.backends.quantized.engine = quant_type
    device = torch.device('cpu')

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

  # define models
  model_list = load_models(args.model.fusion,quantized,device,args.train.n_class)

  
  # load the models
  for i, model in enumerate(model_list):
    model.eval()
    if quantized:
      model_name = os.path.join(path_model,args.result.name+'_'+str(i)+'_quantized.pt')
    else:
      model_name = os.path.join(path_model,args.result.name+'_'+str(i)+'_last.pt')
    print(f"Loading model {i} from {model_name}")
    model.load_state_dict(torch.load(model_name, map_location=device))

  trainer = Trainer(model_list=model_list, 
                  data_train=data_train, 
                  data_valid=[],
                  data_test=data_test,
                  args=args, 
                  device=device,
                  )
  loss_fn=torch.nn.CrossEntropyLoss().to(device)
  test_acc, test_loss, test_y, test_y_pred, test_y_prob, test_des = trainer.test(data_test, device, model_list, loss_fn, 0)

  assert len(test_y) == len(test_y_pred), "Mismatched lengths!"

  # Create a DataFrame
  df = pd.DataFrame({
      'experiment #': range(1, len(test_y) + 1),
      'test_prediction': test_y_pred,
      'test_real': test_y
  })

  df.to_csv(config['csv_save'], index=False)

  matches = sum(a == b for a, b in zip(test_y_pred, test_y))

  print(f"Prob of matching elements: {matches/len(test_y_pred)}")
  print("Length of dataset: ", len(test_y_pred))

  # save_result_confusion(test_y, test_y_pred, trainer.label, 'test-confusion', config['path_save'])

if __name__ == '__main__':
  main()