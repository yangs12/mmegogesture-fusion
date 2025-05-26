import os
import re
import time
import torch
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms
from model import main_Net
from utils.dataloader import ToCHWTensor
from utils.transform import NormalizeTensor

# Load configuration
args = OmegaConf.load("conf/rpi_inf.yaml")
path_model = args.path_model
sensor = [args.sensor.select] if isinstance(args.sensor.select, str) else args.sensor.select

# Collect unique episode files
unique_episodes = {file for file in os.listdir(args.data_dir) if file.endswith('-img.npy')}
print(f"Unique files: {len(unique_episodes)}")
print(args.path_model)

# Load model
device = torch.device('cpu')
model = main_Net.MyNet_Main(args, device).to(device)
model.load_state_dict(torch.load(path_model, map_location=device))
model.eval()

# Build preprocessing transforms
transform = transforms.Compose([
    ToCHWTensor(apply=sensor),
    NormalizeTensor(mean_std=args.transforms.mean_std, apply=sensor)
])

correct_samples = 0
times = []

for episode in sorted(unique_episodes):
    start = time.time()
    match = re.search(r'Class(\d+)', episode)
    label = int(match.group(1)) if match else -1

    episode_base = episode.replace('-img.npy', '')
    data_img = np.load(os.path.join(args.data_dir, f'{episode_base}-img.npy'))
    data_img = np.transpose(data_img, (2, 0, 1))  # (C, H, W)
    data_uD = np.load(os.path.join(args.data_dir, f'{episode_base}-rad-uD.npy'))

    data = {
        'cam-img': data_img,
        'rad-uD': data_uD,
        'label': 0,
        'des': np.nan
    }

    sample = transform(data)
    x_batch = {sensor_sel: sample[sensor_sel].unsqueeze(0).to(device, dtype=torch.float) for sensor_sel in sensor}

    with torch.no_grad():
        y_batch_prob = model(x_batch)
        y_batch_pred = torch.argmax(y_batch_prob, axis=1)
        if y_batch_pred.item() == label:
            correct_samples += 1
        else:
            print(f'Episode {episode_base}: Predicted {y_batch_pred.item()}, Correct {label}')
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f"Average time per episode: {avg_time:.4f} seconds")
print(f"Average accuracy: {correct_samples / len(unique_episodes):.4f}")
