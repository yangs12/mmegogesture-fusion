import torch
from omegaconf import OmegaConf
from model import main_Net  # or mobileVit.main_Net if using MobileViT
from utils.dataloader import ToCHWTensor, CenterCrop_Time, ResampleVideo, CropDoppler, ResizeTensor, LoadDataset_Gesture
from utils.transform import NormalizeTensor
import os
import numpy as np
from torchvision import transforms
# Load configuration
args = OmegaConf.load("conf/rpi_inf.yaml")  # path to your config
path_model = args.path_model
sensor = [args.sensor.select] if isinstance(args.sensor.select, str) else args.sensor.select

data_dir =  '/home/shuboy/captured_data/capture_resized/'
# '/home/shuboy/Desktop/Gesture_data/paper_data/'
# '/home/shuboy/captured_data/capture_resized/'
# Get unique file names with pattern *-img.npy
unique_episodes = set()
for file in os.listdir(data_dir):
    if file.endswith('-img.npy'):
        unique_episodes.add(file)

print(f"Unique files: {unique_episodes}")
# episode = list(unique_episodes)[0].replace('-img.npy', '')

for episode in unique_episodes:
    episode = episode.replace('-img.npy', '')
    print(f"------Episode: {episode}")
    data_img = np.load(os.path.join(data_dir, f'{episode}-img.npy'))
    data_img = np.transpose(data_img, (2, 0, 1))  # Reorder dimensions from (H, W, C) to (C, H, W)
    data_uD = np.load(os.path.join(data_dir,f'{episode}-rad-uD.npy'))
    print(f"Data shapes: {data_img.shape}, {data_uD.shape}")
    data = {}
    data['cam-img'] = data_img
    data['rad-uD'] = data_uD
    data['label'] = 0
    data['des'] = np.nan


    # Build preprocessing transforms (test only)
    transform_list = [
        ToCHWTensor(apply=sensor),
        # CenterCrop_Time(win_snapshot=(args.sensor.win_cam, args.sensor.win_rad),
        #                 t_snapshot=args.sensor.t_snapshot, 
        #                 t_actual=args.transforms.t_actual, 
        #                 apply=sensor),
        # ResampleVideo(resample=args.transforms.resample_cam, apply=sensor),  # RGB only
        # CropDoppler(ratio=args.transforms.cropratio_uD, apply=sensor),       # radar-uD only
        # ResizeTensor(size_rad=(args.transforms.size_rad_D, args.transforms.size_rad_T), 
                    #  size_cam=args.transforms.size_cam, 
                    #  apply=sensor),
        NormalizeTensor(mean_std=args.transforms.mean_std, apply=sensor)
    ]
    transform = transforms.Compose(transform_list)
    sample = transform(data)

    # label = int(gesture.split('e')[-1])-1

    # # Load the dataset to get one sample (or manually prepare one)
    # _, test_dataset = LoadDataset_Gesture(args, {'test': transform_list, 'train': []})
    # sample = test_dataset.dataset[0]
    # print(f"Sample shape: {sample[0].shape, sample[1].shape, sample[2], sample[3]}") # uD 256*512, img 3*256*256, label, des


    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fp32 = main_Net.MyNet_Main(args, device).to(device)
    model_fp32.load_state_dict(torch.load(path_model, map_location=device))
    model_fp32.eval()


    x_batch = {}
    print('sensor:', sensor)
    for sensor_idx, sensor_sel in enumerate(sensor):
        x_batch[sensor_sel] = sample[sensor_sel].unsqueeze(0).to(device, dtype=torch.float)
        
    # Run inference
    with torch.no_grad():
        # y_batch_prob = model_int8(x_batch)
        y_batch_prob = model_fp32(x_batch)
        y_batch_pred = torch.argmax(y_batch_prob, axis=1)
        print(f'\n --------Predicted class: {y_batch_pred.item()}')
