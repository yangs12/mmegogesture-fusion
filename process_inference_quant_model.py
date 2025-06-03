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
from utils.trainer_quant import Trainer
from utils.dataloader import ToCHWTensor, CenterCrop_Time, ResampleVideo, CropDoppler, ResizeTensor, LoadDataset_Gesture
from utils.transform import NormalizeTensor
from torchvision import transforms
import re
from torch.ao.nn.quantized import Linear as nnq_Linear
from torchvision.models.quantization import mobilenet_v2

@hydra.main(version_base="1.2", config_path="conf", config_name="process_inference_quant_model")
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
    
    # ------ Load the quantized model ------
    quant_type = 'qnnpack'
    torch.backends.quantized.engine = quant_type

    # define models
    model_list = []
    if 'concat' in args.model.fusion:
        model1 = mobilenet_v2(pretrained=True, quantize=True)
        model1.classifier = nn.Identity()
        model1.to(device)

        model2 = mobilenet_v2(pretrained=True, quantize=True)
        model2.classifier = nn.Identity()
        model2.to(device)
        
        fusion_classifier = FusionClassifierOptions(1280 * 2, args.train.n_class,dropout=True, batchnorm=True).to(device)
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
    elif 'camonly' in args.model.fusion:
        model1 = mobilenet_v2(pretrained=True, quantize=True)
        model1.classifier = nn.Identity()
        model1.to(device)

        fusion_classifier = FusionClassifierOptions(1280, args.train.n_class,dropout=True, batchnorm=True).to(device)
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
    elif 'late' in args.model.fusion:
        model1 = mobilenet_v2(pretrained=True, quantize=True)
        model1.classifier[1] = nnq_Linear(model1.last_channel, args.train.n_class)
        model1.to(device)

        model2 = mobilenet_v2(pretrained=True, quantize=True)
        model2.classifier[1] = nnq_Linear(model2.last_channel, args.train.n_class)
        model2.to(device)

        model_list = [model1, model2]
    loss_fn=torch.nn.CrossEntropyLoss().to(device)
    
    # load the models
    for i, model in enumerate(model_list):
        model.eval()
        model_name = os.path.join(args.result.path_save_model,args.result.name+'_'+str(i)+'_quantized_finetuned.pt')
        print(f"Loading model {i} from {model_name}")
        model.load_state_dict(torch.load(model_name, map_location=device))
    
    transform = transforms.Compose([
        ToCHWTensor(apply=sensor),
        NormalizeTensor(mean_std=args.transforms.mean_std, apply=sensor)
    ])
    csv_des_data = []

    for radar_file_path in sorted(radar_files):
        # ----- Processing each radar file -----
        start_time = time.time()
        radar_file_name = os.path.splitext(os.path.basename(radar_file_path))[0]
        print(f"\n**Processing {radar_file_path}**")

        radar_loader = radarDataLoader(radar_file_path, radar_params)
        radar_cube, pcloud_list, info, num_frames, r_axis, d_axis = radar_loader.load_data()

        video_path = os.path.join(args.process.folder_path, 'camera', f'{radar_file_name}.avi')
        cap = cv2.VideoCapture(video_path)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        capture_time = int(video_frame_count / video_fps) if video_fps > 0 else 0
        print(f"Capture time: {capture_time} seconds")

        radar_cube = radar_cube[:int(capture_time * radar_params['fps'] + 1), ...]
        print("Final radar_cube shape:", radar_cube.shape)

        uD = RD(radar_cube, args, if_stft=True, window=False)
        uD_fps = args.process.uD_fps #int(uD.shape[1] / args.process.capture_time)

        # print(f"uD shape: {uD.shape}, uD_fps: {uD_fps}")
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
                    middle_frame_index = int(center_indices[i] / uD.shape[1] * capture_time * args.process.cam_fps)
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
                    middle_frame_index = int(center_indices[i] / uD.shape[1] * capture_time * args.process.cam_fps)
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
                    # print(f"Video segment shape: {video_segment.shape}")
                    video_save_path = os.path.join(
                        args.process.output_resize_dir, f'{radar_file_name}-{i}-cam.npy'
                    )
                    # print(f"Saving video segment to {video_save_path}")
                    np.save(video_save_path, video_segment)

            x_batch = {
                'cam-img': np.transpose(img_resized, (2, 0, 1)) if img_resized is not None else None,
                'rad-uD': uD_resized if uD_resized is not None else None,
                'label': np.array([0]),
                'des': np.array([np.nan])
            }
            sample = transform(x_batch)
            
            sample['rad-uD'] = sample['rad-uD'].unsqueeze(0).to(device, dtype=torch.float)
            sample['cam-img'] = sample['cam-img'].unsqueeze(0).to(device, dtype=torch.float) if sample['cam-img'] is not None else None
            
            with torch.no_grad():
                trainer = Trainer(model_list=model_list, 
                data_train=[], 
                data_valid=[],
                data_test=[sample],
                args=args, 
                device=device,
                )
            test_y_pred = trainer.test([sample], device, model_list, loss_fn, 0)

            activity = args.process.activity if args.process.activity is not None else test_y_pred
            print(f'--------Segment {i} Predicted class: {test_y_pred} {gesture_map[test_y_pred]}')

            # ----- Save the Data -----            
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
    df.to_csv(os.path.join(args.process.output_dir, 'des_rpi_shubo_May26_test_quant.csv'), index=False)
    print(f"Radar mean: {np.mean(radar_mean)}, Radar std: {np.std(radar_mean)}")
    print(f"Camera mean: {np.mean(cam_mean, axis=0)}, Camera std: {np.std(cam_mean, axis=0)}")
    print(f"Average time taken for all episodes: {sum(times) / len(times)} seconds")

if __name__ == "__main__":
    main()
