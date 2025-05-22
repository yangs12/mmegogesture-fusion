import torch
from omegaconf import OmegaConf
from model import main_Net  # or mobileVit.main_Net if using MobileViT
from utils.dataloader import ToCHWTensor, CenterCrop_Time, ResampleVideo, CropDoppler, ResizeTensor, LoadDataset_Gesture
from utils.transform import NormalizeTensor
import os
import numpy as np
from torchvision import transforms
import time
# Load configuration
args = OmegaConf.load("conf/rpi_inf.yaml")  # path to your config
path_model = args.path_model
sensor = [args.sensor.select] if isinstance(args.sensor.select, str) else args.sensor.select

data_dir =  '/home/shuboy/captured_data/capture_resized/2025May21-1709/' #2025May21-1310/' #2025May21-1229/
# '/home/shuboy/Desktop/Gesture_data/paper_data/'
# '/home/shuboy/captured_data/capture_resized/'
# Get unique file names with pattern *-img.npy
unique_episodes = set()
for file in os.listdir(data_dir):
    if file.endswith('-img.npy'):
        unique_episodes.add(file)

print(f"Unique files: {unique_episodes}")
# episode = list(unique_episodes)[0].replace('-img.npy', '')
times = []
# Load model
device = torch.device('cpu')
model_fp32 = main_Net.MyNet_Main(args, device).to(device)
model_fp32.load_state_dict(torch.load(path_model, map_location=device))
model_fp32.eval()

# Build preprocessing transforms (test only)
transform_list = [
    ToCHWTensor(apply=sensor),
    # CenterCrop_Time(win_snapshot=(args.sensor.win_cam, args.sensor.win_rad),
    #                 t_snapshot=args.sensor.t_snapshot, 
    #                 t_actual=args.transforms.t_actual, 
    #                 apply=sensor),
    # ResizeTensor(size_rad=(args.transforms.size_rad_D, args.transforms.size_rad_T), 
                #  size_cam=args.transforms.size_cam, 
                #  apply=sensor),
    NormalizeTensor(mean_std=args.transforms.mean_std, apply=sensor)
]
transform = transforms.Compose(transform_list)

labels = [i for i in range(12) for _ in range(3)]
episode_i = 0
correct_samples = 0
for episode in sorted(unique_episodes):
    start = time.time()
    episode = episode.replace('-img.npy', '')
    print(f"------Episode: {episode}")
    data_img = np.load(os.path.join(data_dir, f'{episode}-img.npy'))
    data_img = np.transpose(data_img, (2, 0, 1))  # Reorder dimensions from (H, W, C) to (C, H, W)
    data_uD = np.load(os.path.join(data_dir,f'{episode}-rad-uD.npy'))
    # print(f"Data shapes: {data_img.shape}, {data_uD.shape}")
    data = {}
    data['cam-img'] = data_img
    data['rad-uD'] = data_uD
    data['label'] = 0
    data['des'] = np.nan



    sample = transform(data)
    # print(sample['cam-img'].shape, np.mean(sample['cam-img'][0]))


    x_batch = {}
    # print('sensor:', sensor) #sensor: ['rad-uD', 'cam-img']
    for sensor_idx, sensor_sel in enumerate(sensor):
        x_batch[sensor_sel] = sample[sensor_sel].unsqueeze(0).to(device, dtype=torch.float)
        
    # Run inference
    with torch.no_grad():
        # y_batch_prob = model_int8(x_batch)
        y_batch_prob = model_fp32(x_batch)
        y_batch_pred = torch.argmax(y_batch_prob, axis=1)
        if y_batch_pred.item() == labels[episode_i]:
            correct_samples += 1
        print(f'--------Predicted class: {y_batch_pred.item()}')
        episode_i += 1
    end = time.time()
    times.append(end - start)
    print(f"Time taken for episode {episode}: {end - start} seconds \n")
# Calculate and print the average time taken
avg_time = sum(times) / len(times)
print(f"Average time taken for all episodes: {avg_time} seconds")
print("average acc", correct_samples / len(unique_episodes))