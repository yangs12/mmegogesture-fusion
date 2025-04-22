import os
import cv2
import sys
import torch
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import signal


def img2video(img_dir, vid_dir):
    video_name = ('_').join(img_dir.split('/')[-3:-1])
    fps = 30.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    names = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])
    videoWrite = cv2.VideoWriter(vid_dir, fourcc, fps, size) 
    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)
    videoWrite.release()
    return



def show_all(vid, data_GT, data_Pred, data_startpoint, select_frame, path=None):
    #
    path_folder = ('/').join(path.split('/')[:-1])
    path_temp   = os.path.join(path_folder,'temp/')
    os.makedirs(path_temp, exist_ok=True)
    t_idx = np.linspace(data_startpoint[select_frame], data_startpoint[select_frame]+90-1, 90)
    # Keypoint
    output_GT   = data_GT[select_frame,:,:,:]
    output_Pred = data_Pred[select_frame,:,:,:]
    output_GT   = resize_keypoint(output_GT, 90)
    output_Pred = resize_keypoint(output_Pred, 90)
    for i in range(len(t_idx)):
        img       = vid[int(t_idx[i])]
        pose_GT   = output_GT[i]
        pose_Pred = output_Pred[i]
        ## show
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax2 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3 = fig.add_subplot(gs[0, 0])
        show3Dpose(pose_GT, ax1, save=False)
        show3Dpose(pose_Pred, ax2, save=False)
        showimage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ax3)
        ## save
        plt.savefig(path_temp + str(('%04d'% i)) + '_pose.png', dpi=300, bbox_inches = 'tight')
        fig.tight_layout()
        plt.close()
    img2video(path_temp, path)
    shutil.rmtree(path_temp)
    return

def showRGB_video(vid, path=None, name=None):
    fps = 30.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (640, 480)
    os.makedirs(path, exist_ok=True)
    videoWrite = cv2.VideoWriter(path+name, fourcc, fps, size)
    for i in range(len(vid)):
        videoWrite.write(cv2.resize(vid[i], size))
    videoWrite.release()
    return

def show_radar(rad, path=None, name=None):
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(rad[:,:,0],cmap='jet',aspect=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path+name, bbox_inches='tight')
    plt.close()
    return

def show_RGB_image(vid, path=None, name=None):
    len_vid = len(vid)
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(vid[len_vid//2,:,:,:], cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path+name, bbox_inches='tight')
    plt.close()
    return	


def showimage(img, ax):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)

def show3Dpose(vals_pre, ax, path=None, save=True):
    ax.view_init(elev=15., azim=70)

    vals_pre = np.array(vals_pre, dtype='float32')
    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    vals = camera_to_world(vals_pre, R=rot, t=0)
    # vals = np.array(vals_pre)

    lcolor=(1,0,0)  # Strange.. Why is this color reversed with 2D? (TODO)
    rcolor=(0,0,1)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)
    # test_idx = 1
    # ax.scatter(vals[test_idx,0],vals[test_idx,1],vals[test_idx,2], 'o', color=(0,1,0))
    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    # remove tick labels
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    # ax.tick_params('x', labelbottom = True)
    # ax.tick_params('y', labelleft = True)
    # ax.tick_params('z', labelleft = True)
    if save==True:
        plt.savefig(path, dpi=400, bbox_inches = 'tight')

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result): 
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)



        