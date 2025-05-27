import os
import sys
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
import torch.quantization
from torchvision.models.quantization import mobilenet_v2
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import einops
from utils.dataloader import *
from utils.trainer_quant import Trainer
from model.classifier_head import FusionClassifier


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def inspect_model_quantization(model, model_name="model"):
    print(f"\n🔍 Inspecting {model_name}...\n")

    # 1. Check if any quantized modules are used
    print("=== Layer types ===")
    for name, module in model.named_modules():
        if 'Quant' in str(type(module)) or 'quantized' in str(type(module)).lower():
            print(f"{name}: {type(module)}")

    # 2. Check parameter data types
    print("\n=== Parameter data types ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")


def run_model_list(x_batch,fusion_method,model_list):
    if fusion_method == 'concat':
        radar_processed_data = einops.repeat(x_batch['rad-uD'], 'b h w -> b (copy) h w', copy=3)
        feat1 = model_list[0](radar_processed_data)
        feat2 = model_list[1](x_batch['cam-img'])
        fused = torch.cat([feat1, feat2], dim=1)
        y_batch_pred = model_list[-1](fused)

    elif fusion_method == 'camonly':
        feat1 = model_list[0](x_batch['cam-img'])
        y_batch_pred = model_list[-1](feat1)

    elif fusion_method == 'late':
        radar_processed_data = einops.repeat(x_batch['rad-uD'], 'b h w -> b (copy) h w', copy=3)
        output1 = model_list[0](radar_processed_data)
        output2 = model_list[1](x_batch['cam-img'])
        y_batch_pred = (output1 + output2) / 2.0

    return y_batch_pred




@hydra.main(version_base=None,  config_path="conf/quant/camonly_config", config_name="camonly_678")
def main(args: DictConfig) -> None:
    config = OmegaConf.to_container(args)
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

    quant_type = 'qnnpack' # or fbgemm or qnnpack
    torch.backends.quantized.engine = quant_type

    # ----- STEP 1: Load and prepare float32 model -----
    model_list = []
    if 'concat' in args.model.fusion:
        model1 = mobilenet_v2(pretrained=True, quantize=False)
        model1.classifier = nn.Identity()
        model1.to(device)

        model2 = mobilenet_v2(pretrained=True, quantize=False)
        model2.classifier = nn.Identity()
        model2.to(device)

        fusion_classifier = FusionClassifier(1280 * 2, args.train.n_class).to(device)

        model_list = [model1, model2, fusion_classifier]
    elif 'camonly' in args.model.fusion:
        model1 = mobilenet_v2(pretrained=True, quantize=False)
        model1.classifier = nn.Identity()
        model1.to(device)

        fusion_classifier = FusionClassifier(1280, args.train.n_class).to(device)

        model_list = [model1, fusion_classifier]
    elif 'late' in args.model.fusion:
        model1 = mobilenet_v2(pretrained=True, quantize=False)
        model1.classifier[1] = nn.Linear(model1.last_channel, args.train.n_class)
        model1.to(device)

        model2 = mobilenet_v2(pretrained=True, quantize=False)
        model2.classifier[1] = nn.Linear(model2.last_channel, args.train.n_class)
        model2.to(device)

        model_list = [model1, model2]

    # load the models
    for i, model in enumerate(model_list):
        model_name = os.path.join(args.result.path_save_model,args.result.name+'_'+str(i)+'_last.pt')
        model.load_state_dict(torch.load(model_name, map_location=device))

        model.eval()
        model.to(device)

        if hasattr(model, 'fuse_model'):
            model.fuse_model()  # Fuse layers if applicable


        # setup quantization config and prepare
        model.qconfig = torch.quantization.get_default_qconfig(quant_type)
        torch.quantization.prepare(model, inplace=True)

    # load trainer
    trainer = Trainer(model_list=model_list, 
                    data_train=data_train, 
                    data_valid=[],
                    data_test=data_test,
                    args=args, 
                    device=device
                    )
    

    # Calibration step
    print("🔍 Running calibration step...")
    for model in model_list:
        model.eval()
    trainer.calibrate_model(total_batches = 20)  # This will run the calibration step

    # print observers
    # print("\n=== Observers in each model ===")
    # for i, model in enumerate(model_list):
    #     print(f"\n=== MODEL {i} ===")
    #     for name, module in model.named_modules():
    #         if isinstance(module, torch.ao.quantization.observer.ObserverBase):
    #             print(f"{name} → {type(module).__name__}")

    # # ----- STEP 4: Convert to quantized model -----
    for i, model in enumerate(model_list):
        torch.quantization.convert(model, inplace=True)
        # inspect_model_quantization(model, model_name='model' + str(i))
        model_name = os.path.join(args.result.path_save_model,args.result.name+'_'+str(i)+'_quantized.pt')
        torch.save(model.state_dict(), model_name)

    
    # inference
    for model in model_list:
        model.eval()

    # ----- STEP 7: Inference test ----
    loss_fn=torch.nn.CrossEntropyLoss().to(device)
    test_acc, test_loss, test_y, test_y_pred, test_y_prob, test_des = trainer.test(data_test, device, model_list, loss_fn, 0)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
