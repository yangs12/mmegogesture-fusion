from .radar_config import RadarConfig
from .radar_capture_utils import DCA1000
import numpy as np
# import moviepy.editor as mp
import pandas as pd
# from .mmwave.tracking import gtrack_visualize
import cv2
import os
import shutil
from tqdm import tqdm
import json
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

# classes
class radarDataLoader:
    '''
        Load radar data from npz file
    '''

    def __init__(self, file_path, radar_params):
        self.file_path = file_path
        self.params = radar_params
    
    def load_data(self):
        data_npz = np.load(self.file_path)
        radar_data_uint8 = data_npz['radar_data']

        num_Rx = int(self.params['n_rx'])
        num_Tx = int(self.params['n_tx'])
        radar_frame, pcloud_list, info, num_frames = DCA1000.decode_data(radar_data_uint8, num_Tx=num_Tx, num_Rx=num_Rx)

        n_chirps = int(self.params['n_chirps'])
        n_samples = int(self.params['n_samples'])
        range_res = float(self.params['range_res'])
        velocity_res = float(self.params['velocity_res'])

        range_axis = np.arange(n_samples)*range_res
        doppler_axis = np.arange(-n_chirps//2, n_chirps//2)*velocity_res
        return radar_frame, pcloud_list, info, num_frames, range_axis, doppler_axis

# functions
def pw2db(x, scale=10):
    return scale*np.log10(np.abs(x)+1e-9)

def calc_angle_to_y(loc):
    return np.abs(np.rad2deg(np.arctan(loc[0]/loc[1])))


def RD(rda, args=None, declutter=True, window=True, track_info=None, if_stft=False, r_axis=None):
    '''
        Perform range and doppler FFT on radar_cube data
        Input:
        - radar_cube: radar data in time domain
            - shape: (Nframes, Nchirps, Ntx, Nrx, Nsamples)
        - declutter: remove dc component or not
        - window: apply windowing or not
        Output:
        - RD: radar data after range and doppler FFT
            - shape: (Nframes, Nchirps, Ntx*Nrx, Nsamples)
    '''
    # Decluttering in range dimension
    if declutter:
        rda -= np.expand_dims(rda.mean(axis=1), axis=1) # remove dc clutter
    if window:
        window_1d_range = np.hanning(rda.shape[-1])
        window_1d_doppler = np.hanning(rda.shape[1])
        rda = rda * window_1d_range[None, None, None, None, :]
        rda = rda * window_1d_doppler[None, :, None, None, None]
    

    if track_info is not None and r_axis is not None:
        loc_x = -np.array(track_info["loc_x"])
        loc_y = np.array(track_info["loc_y"])
        
        rda_bbox = []
        # if wanting a tqdm progress bar:
        # for i, frame in enumerate(tqdm(track_info["frames"], total=len(track_info["frames"]), desc="Processing frames")):
        for i, frame in enumerate(track_info["frames"]):
            loc_x_i = loc_x[i]; loc_y_i = loc_y[i]
            (r_bounds, angle_bounds) = xycenter2bbox(loc_x_i, loc_y_i, r_axis, args)
            # TODO: maybe later we need to filter out people too close to radar, since they wont have enough ridxs

            rda_i = rda[frame, :, :, :, :] # shape: (Nchirps, Ntx, Nrx, Nsamples).
            Rda_i = np.fft.fft(rda_i, axis=3) #/ np.sqrt(rda_i.shape[3]) # shape: (Nchirps, Ntx, Nrx, Nsamples)
            Rda_i = Rda_i.reshape(Rda_i.shape[0], Rda_i.shape[1]*Rda_i.shape[2], Rda_i.shape[3]) # shape: (Nchirps, Ntx*Nrx, Nsamples)
            
            # # Antenna pair phase offset correction
            # for idx, offset in enumerate(args.virtual_ant_phase_offsets, start=1):
            #     Rda_i[:, idx, :] *= np.exp(1j * offset)

            RdA_i = np.fft.fftshift(np.fft.fft(Rda_i, axis=1, n=args.n_angle_fft), axes=1) #/ np.sqrt(Rda_i.shape[1]) # shape: (Nchirps, NangleFFT, Nsamples))
            RdA_bbox_i = RdA_i[:, angle_bounds[0]:angle_bounds[1]+1, r_bounds[0]:r_bounds[1]+1]
            rdA_bbox_i = np.fft.ifft(RdA_bbox_i, axis=2)
            rda_bbox_i = np.fft.ifft(np.fft.ifftshift(rdA_bbox_i, axes=1), axis=1) 
            rda_bbox.append(rda_bbox_i)

        if if_stft:
            # Collapse the angle dimension, since at different range, angle FFT might have different number of bins
            rd_bbox = [np.mean(frame, axis=1) for frame in rda_bbox]
            rd_bbox = np.stack(rd_bbox, axis=0) # shape: (Nframes_of_tracks, Nchirps, range bbox)

            # STFT
            x = rd_bbox.reshape((-1, rd_bbox.shape[-1])) # collapse all dimensions except for the last one - range, only work for 1 TxRx pair            
            f, t, Zxx = stft(x, nfft=args.n_uD_fft,nperseg=args.n_uD_fft,noverlap=int(args.overlap_ratio*args.n_uD_fft),window=args.uD_window,return_onesided=False,axis=0)
            Zxx=Zxx.transpose((0,2,1))
            Zxx=np.fft.fftshift(Zxx,0)
            uD=10*np.log(np.mean(np.abs(Zxx),-1) + 1e-9) 
            return uD  
        else:
            return rda_bbox # list of rda_bbox_i (cropped rda data)

    elif if_stft:
        dcube_uD = rda.reshape((-1, 3,4,rda.shape[-1]))
        # STFT
        # x = rd_bbox.reshape((-1, rd_bbox.shape[-1])) # collapse all dimensions except for the last one - range, only work for 1 TxRx pair
        f, t, Zxx = stft(dcube_uD, nfft=args.process.n_uD_fft,nperseg=args.process.stft_window_size,noverlap=int(args.process.overlap_ratio*args.process.stft_window_size),window=args.process.uD_window,return_onesided=False,axis=0)
        # Zxx=Zxx.transpose((0,2,1))
        Zxx=np.fft.fftshift(Zxx,0)
        uD = 20*np.log10(np.abs(Zxx).mean((1,2,3)) + 1e-9)
        # uD=10*np.log(np.mean(np.abs(Zxx),-1) + 1e-9) 
        return uD  
        
    # No tracking and no cropping bbox, only FFT for RDa
    else:
        RDa = np.fft.fftshift(np.fft.fft(rda, axis=1), axes=1) #/ np.sqrt(rda.shape[1])
        RDa = np.fft.fft(RDa, axis=4) #/ np.sqrt(RDa.shape[4])

        # from (Nf, Nc, Ntx, Nrx, Nsamples) to (Nf, Nc, Ntx*Nrx, Nsamples)
        RDa = RDa.reshape(RDa.shape[0], RDa.shape[1], RDa.shape[2]*RDa.shape[3], RDa.shape[4])
        # RDa = RDa[..., :8, :] # only keep the first 8 pairs
        print(f"RDa shape: {RDa.shape}")
        noise_floor_db = 10*np.log10(np.mean(np.abs(RDa.mean(axis = 2))**2))
        noise_range_db = 10*np.log10(np.mean(np.abs(RDa.mean(axis = 2))**2, axis=(0,1)))

        return RDa, noise_floor_db, noise_range_db

def save_uD_plot(uD, args, radar_file_name, uD_axis, track_id=None):
    # Saving the plot
    fig_width = max(6, uD.shape[1] / args.plot_scale)  # adjust scaling factor as needed
    # fig, ax = plt.subplots(figsize=(fig_width, 6))
    fig, ax = plt.subplots(figsize=(10, 6))  # Set dpi to 300 for high resolution

    # Plot the micro-Doppler image with adjustable aspect ratio
    im = ax.imshow(uD, aspect='auto', cmap='jet', vmax=args.vmax, vmin=args.vmin)
    fig.colorbar(im, ax=ax)

    if track_id is not None:
        ax.set_title(f'{radar_file_name} track ID {track_id} micro-Doppler')
    else:
        ax.set_title(f'{radar_file_name} micro-Doppler')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Doppler (m/s)')
    # ticks = np.linspace(0, uD.shape[1], 11)
    # tick_labels = np.round(ticks / args.uD_bins_ps, 2)
    # plt.xticks(ticks, tick_labels)
    # plt.yticks(np.linspace(0, args.n_uD_fft, 11), np.round(np.linspace(uD_axis[0], uD_axis[-1], 11), 2))
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0.5)
    if track_id is not None:
        plt.savefig(os.path.join(args.output_dir, 'tracks_uD_figs', f'{radar_file_name}_track_{track_id}.png'))
    else:
        plt.savefig(os.path.join(args.output_dir, f'{radar_file_name}.png'))
    plt.close()


def create_video(frames, path, fps=10):
    clips = [mp.ImageClip(frame).set_duration(1/fps) for frame in frames]
    video = mp.concatenate_videoclips(clips, method="compose")
    video.write_videofile(path, fps=fps)

def gesture_detection(uD, args, uD_fps):
    # Step 1: Get indices over threshold
    gesture_idxs = np.where(uD.mean(0) > args.process.uD_threshold)[0]
    if len(gesture_idxs) == 0:
        return np.array([]), np.array([]), []

    # Step 2: Group into segments allowing for small gaps
    gesture_segments = []
    start = gesture_idxs[0]
    for i in range(1, len(gesture_idxs)):
        if gesture_idxs[i] - gesture_idxs[i - 1] > args.process.gesture_gap_thresh * uD_fps:
            end = gesture_idxs[i - 1]
            gesture_segments.append((start, end))
            start = gesture_idxs[i]
    gesture_segments.append((start, gesture_idxs[-1]))

    # Filter by length
    min_len = args.process.gesture_min_len * uD_fps
    max_len = args.process.gesture_max_len * uD_fps
    gesture_segments = np.array([
        (s, e) for s, e in gesture_segments if (e - s + 1) >= min_len and (e - s + 1) <= max_len
    ])

    center_indices = np.array([int((s + e) / 2) for s, e in gesture_segments])
    print("Number of gestures detected:", len(gesture_segments), "Center indices (s):", center_indices / uD_fps)

    # For each center index, get a segment centered around it
    center_segments = []
    seg_len = int(args.process.center_segment_second * uD_fps)
    half_seg = seg_len // 2
    for c in center_indices:
        start_idx = max(0, c - half_seg)
        end_idx = min(c + half_seg, uD.shape[1])
        center_segments.append([int(start_idx), int(end_idx)])

    return gesture_segments, center_indices, center_segments