import torch
from omegaconf import OmegaConf
from model import main_Net  # or mobileVit.main_Net if using MobileViT
from utils.dataloader import ToCHWTensor, CenterCrop_Time, ResampleVideo, CropDoppler, ResizeTensor, LoadDataset_Gesture

# Load configuration
args = OmegaConf.load("conf/rpi_inf.yaml")  # path to your config
path_model = args.path_model
sensor = [args.sensor.select] if isinstance(args.sensor.select, str) else args.sensor.select

# Build preprocessing transforms (test only)
transform_list = [
    ToCHWTensor(apply=sensor),
    CenterCrop_Time(win_snapshot=(args.sensor.win_cam, args.sensor.win_rad),
                    t_snapshot=args.sensor.t_snapshot, 
                    t_actual=args.transforms.t_actual, 
                    apply=sensor),
    ResampleVideo(resample=args.transforms.resample_cam, apply=sensor),  # RGB only
    CropDoppler(ratio=args.transforms.cropratio_uD, apply=sensor),       # radar-uD only
    ResizeTensor(size_rad=(args.transforms.size_rad_D, args.transforms.size_rad_T), 
                 size_cam=args.transforms.size_cam, 
                 apply=sensor)
]

# Load the dataset to get one sample (or manually prepare one)
_, test_dataset = LoadDataset_Gesture(args, {'test': transform_list, 'train': []})
sample = test_dataset.dataset[0]


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = main_Net.MyNet_Main(args, device).to(device)
model.load_state_dict(torch.load(path_model, map_location=device))
model.eval()

# Prepare inputs
x_batch = {}
for sensor_idx, sensor_sel in enumerate(sensor):
    x_batch[sensor_sel] = sample[sensor_idx].unsqueeze(0).to(device, dtype=torch.float)

# Run inference
with torch.no_grad():
    y_batch_prob = model(x_batch)
    y_batch_pred = torch.argmax(y_batch_prob, axis=1)
    print(f'Predicted class: {y_batch_pred.item()}')
