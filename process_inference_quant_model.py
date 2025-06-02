from utils.radar_config import RadarConfig
from utils.pipeline_utils import *
import numpy as np
from tqdm import tqdm
import os
import hydra
from omegaconf.dictconfig import DictConfig
import cv2
import pandas as pd
import json
from omegaconf import OmegaConf
import time
import torch
from model import main_Net
from utils.dataloader import ToCHWTensor, CenterCrop_Time, ResampleVideo, CropDoppler, ResizeTensor, LoadDataset_Gesture
from utils.transform import NormalizeTensor
from torchvision import transforms
import re

@hydra.main(version_base="1.2", config_path="conf", config_name="process_inference")
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)

    os.makedirs(args.process.output_resize_dir, exist_ok=True)
    os.makedirs(args.process.output_dir, exist_ok=True)

    radar_mean, radar_std, cam_mean, cam_std, times = [], [], [], [], []

    file_date = args.process.radar_data_path.split("/")[-3].split(".")[0] + "/"
    print(f"Processing {file_date} folder")

    if args.process.radar_file_name:
        radar_files = [os.path.join(args.process.radar_data_path, f"{args.process.radar_file_name}.npz")]
    else:
        radar_files = [
            os.path.join(args.process.radar_data_path, f)
            for f in os.listdir(args.process.radar_data_path)
            if f.endswith('.npz')
        ]

    with open(args.process.radar_config_path, 'r') as f:
        cfg = f.readlines()
    radar_config = RadarConfig(cfg)
    radar_params = radar_config.get_params()

    uD_axis = np.arange(-args.process.n_uD_fft // 2, args.process.n_uD_fft // 2) * 2 * radar_params['velocity_max'] / args.process.n_uD_fft

    gesture_map = {
        0: 'Click', 1: 'FingerSnap', 2: 'ZoomIn', 3: 'Knock', 4: 'SwipeLeft',
        5: 'SwipeRight', 6: 'Twohand_ZoomIn', 7: 'Twohand_ZoomOut', 8: 'Stop',
        9: 'OK', 10: 'ThumbUp', 11: 'Photo'
    }
    sensor = [args.sensor.select] if isinstance(args.sensor.select, str) else args.sensor.select
    device = torch.device('cpu')
    model_fp32 = main_Net.MyNet_Main(args, device).to(device)
    model_fp32.load_state_dict(torch.load(args.path_model, map_location=device))
    model_fp32.eval()
    transform = transforms.Compose([
        ToCHWTensor(apply=sensor),
        NormalizeTensor(mean_std=args.transforms.mean_std, apply=sensor)
    ])
    csv_des_data = []

    for radar_file_path in sorted(radar_files):
        start_time = time.time()
        radar_file_name = os.path.splitext(os.path.basename(radar_file_path))[0]
        print(f"\n**Processing {radar_file_path}**")

        radar_loader = radarDataLoader(radar_file_path, radar_params)
        radar_cube, pcloud_list, info, num_frames, r_axis, d_axis = radar_loader.load_data()
        radar_cube = radar_cube[:int(args.process.capture_time * radar_params['fps'] + 1), ...]
        print("Final radar_cube shape:", radar_cube.shape)

        uD = RD(radar_cube, args, if_stft=True, window=False)
        uD_fps = int(uD.shape[1] / args.process.capture_time)
        uD_fps = 240.0
        print(f"uD shape: {uD.shape}, uD_fps: {uD_fps}")
        gesture_segments, center_indices, center_segments, center_segments_videos = gesture_detection(uD, args, uD_fps, radar_file_name)

        if args.process.if_visualize_segments:
            visualize_segments(uD, center_segments, center_indices, radar_file_name, args)
        
        
        for i, (start_idx, end_idx) in enumerate(center_segments):
            if (end_idx - start_idx) / uD_fps < args.process.min_segment_length:
                print(f"Skipping segment {i} due to insufficient length: {(end_idx - start_idx) / uD_fps} < {args.process.min_segment_length}")
                continue
            uD_segment = uD[:, start_idx:end_idx]
            uD_resized, img_resized = None, None
            if args.process.if_resize_uD:
                uD_resized = cv2.resize(
                    uD_segment, (args.process.uD_width, args.process.uD_height), interpolation=cv2.INTER_LINEAR
                )
                radar_mean.append(np.mean(uD_resized))
                radar_std.append(np.std(uD_resized))
            if args.process.if_resize_img:
                video_path = os.path.join(args.process.folder_path, 'camera', f'{radar_file_name}.avi')
                frames = []
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                if frames:
                    middle_frame_index = int(center_indices[i] / uD.shape[1] * args.process.capture_time * args.process.cam_fps)
                    middle_frame_index = min(middle_frame_index, len(frames) - 1)
                    img = frames[middle_frame_index]
                    img_resized = cv2.resize(img, (args.process.img_width, args.process.img_height), interpolation=cv2.INTER_LINEAR)
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    cam_mean.append(np.mean(img_resized, axis=(0, 1)))
                    cam_std.append(np.std(img_resized, axis=(0, 1)))
            if args.process.if_save_video: 
                video_path = os.path.join(args.process.folder_path, 'camera', f'{radar_file_name}.avi')
                frames = []
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                if frames:
                    middle_frame_index = int(center_indices[i] / uD.shape[1] * args.process.capture_time * args.process.cam_fps)
                    middle_frame_index = min(middle_frame_index, len(frames) - 1)
                    start_frame = max(0, middle_frame_index - int(args.process.video_segment_second / 2 * args.process.cam_fps))
                    end_frame = min(len(frames), middle_frame_index + int(args.process.video_segment_second / 2 * args.process.cam_fps))
                    video_segment = frames[start_frame:end_frame]
                    video_segment = [cv2.resize(frame, (args.process.img_width, args.process.img_height), interpolation=cv2.INTER_LINEAR) for frame in video_segment]
                    video_segment = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video_segment]
                    # Pad video_segment with zeros if it has less frames than expected
                    expected_frames = int(args.process.video_segment_second * args.process.cam_fps)
                    if len(video_segment) < expected_frames and len(video_segment) > 0:
                        h, w, c = video_segment[0].shape
                        pad_frames = expected_frames - len(video_segment)
                        zero_frames = [np.zeros((h, w, c), dtype=video_segment[0].dtype) for _ in range(pad_frames)]
                        video_segment.extend(zero_frames)
                    video_segment = np.stack(video_segment, axis=0)
                    print(f"Video segment shape: {video_segment.shape}")
                    video_save_path = os.path.join(
                        args.process.output_resize_dir, f'{radar_file_name}-{i}-cam.npy'
                    )
                    print(f"Saving video segment to {video_save_path}")
                    np.save(video_save_path, video_segment)

            data = {
                'cam-img': img_resized.transpose(2, 0, 1) if img_resized is not None else None,
                'rad-uD': uD_resized,
                'label': 0,
                'des': np.nan
            }
            sample = transform(data)
            x_batch = {sensor_sel: sample[sensor_sel].unsqueeze(0).to(device, dtype=torch.float) for sensor_sel in sensor}

            with torch.no_grad():
                y_batch_prob = model_fp32(x_batch)
                y_batch_pred = torch.argmax(y_batch_prob, axis=1)
                activity = args.process.activity if args.process.activity is not None else y_batch_pred.item()
                print(f'--------Segment {i} Predicted class: {y_batch_pred.item()} {gesture_map[y_batch_pred.item()]}')


            
            if args.process.if_save_uD:
                if args.process.if_resize_uD:
                    np.save(
                        os.path.join(args.process.output_resize_dir, f'{radar_file_name}-{i}-rad-uD.npy'),
                        uD_resized
                    )
                    save_uD_plot(
                        uD_resized, args,
                        f'{radar_file_name}-{i}_resized',
                        uD_axis
                    )
                else:
                    np.save(
                        os.path.join(args.process.output_dir, f'{radar_file_name}-{i}-rad-uD.npy'),
                        uD_segment
                    )
                    save_uD_plot(
                        uD_segment, args,
                        f'{radar_file_name}-{i}',
                        uD_axis
                    )

            if args.process.if_save_img:
                if args.process.if_resize_img:
                    np.save(
                        os.path.join(args.process.output_resize_dir, f'{radar_file_name}-{i}-img.npy'),
                        img_resized
                    )
                    cv2.imwrite(
                        os.path.join(args.process.output_dir, f'{radar_file_name}-{i}-img.png'),
                        img_resized
                    )
                else:
                    np.save(
                        os.path.join(args.process.output_dir, f'{radar_file_name}-{i}-img.npy'),
                        img
                    )
                    cv2.imwrite(
                        os.path.join(args.process.output_dir, f'{radar_file_name}-{i}-img.png'),
                        img
                    )

            csv_des_data.append([
                radar_file_name, i, f"Subject{args.process.subject}", f"Environment{args.process.environment}", f"Gesture{activity}", "TRUE",
            ])

        print(f"Radar mean: {np.mean(radar_mean)}, Radar std: {np.std(radar_mean)}")
        end_time = time.time()
        if len(gesture_segments) > 0:
            times.append((end_time - start_time) / len(gesture_segments))
            print(f"\n Time taken per seg: {(end_time - start_time)/len(gesture_segments)} seconds")
    
    df = pd.DataFrame(csv_des_data, columns=[
        'Episode', 'Order', 'Subject', 'Enviroment', 'Gesture', 'Remark_Episode'
    ])
    df.reset_index(inplace=True)
    df['Remark_Snapshot'] = ""
    df.to_csv(os.path.join(args.process.output_dir, 'des_rpi_shubo_May26_wvideos.csv'), index=False)
    print(f"Radar mean: {np.mean(radar_mean)}, Radar std: {np.std(radar_mean)}")
    print(f"Camera mean: {np.mean(cam_mean, axis=0)}, Camera std: {np.std(cam_mean, axis=0)}")
    print(f"Average time taken for all episodes: {sum(times) / len(times)} seconds")

if __name__ == "__main__":
    main()
