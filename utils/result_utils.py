import os
import torch
import copy
import sklearn
import pickle
import numpy as np
from scipy import signal
from numpy import linalg as LA
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import random

from pytorchvideo.data.encoded_video import EncodedVideo
from PIL import Image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sensordata_load(dat_info, data_dir,change_channel=False):
    (episode, order, sensor_name) = dat_info
    if 'v2' in data_dir:    # 'v2'
        if 'vid' in sensor_name:
            data_path = os.path.join(data_dir,f'{episode}-{order}-cam.mp4')
            video = EncodedVideo.from_path(data_path)
            dat_vid = video.get_clip(start_sec=0, end_sec=9999)['video']
            return dat_vid
        elif 'img' in sensor_name:
            data_path = os.path.join(data_dir,f'{episode}-{order}-img.jpg')
            return Image.open(data_path)
        else:
            data_path = os.path.join(data_dir,f'{episode}-{order}-{sensor_name}.npy')
            return np.load(data_path)
    else:                   # 'v1'
        if 'vid' in sensor_name:
            data_path = os.path.join(data_dir,f'{episode}-{order}-cam.npy')
            dat_vid = np.load(data_path)
            return dat_vid.transpose(3,0,1,2)
        elif 'img' in sensor_name:
            data_path = os.path.join(data_dir,f'{episode}-{order}-img.npy')
            if change_channel:
                dat_img = np.load(data_path)
                return dat_img.transpose(2,0,1)
            return np.load(data_path)
        else:
            data_path = os.path.join(data_dir,f'{episode}-{order}-{sensor_name}.npy')
            return np.load(data_path)

def split_traintest(des_clean, mode):
    ## random split
    if mode=='random':
        random.Random(22).shuffle(des_clean)
        random.Random(2256).shuffle(des_clean)
        des_train = des_clean[:round(len(des_clean)*0.8)]
        des_test = des_clean[round(len(des_clean)*0.8):]
    ## subject independent split
    elif 'subject' in mode:
        test_subject = mode.split('-')[-1].split('_')
        test_subject = [f'Subject{subject_id}' for subject_id in test_subject]
        des_clean = pd.DataFrame(des_clean)
        for sub_idx in range(len(test_subject)):
            if sub_idx ==0:
                val_train = (des_clean['Subject']!=test_subject[sub_idx])
                val_test  = (des_clean['Subject']==test_subject[sub_idx])
            else:
                val_train = val_train & (des_clean['Subject']!=test_subject[sub_idx])
                val_test  = val_test | (des_clean['Subject']==test_subject[sub_idx])
        des_train = des_clean[val_train].to_dict('records')
        des_test  = des_clean[val_test].to_dict('records')
        assert len(des_clean)==(len(des_train)+len(des_test))
    ## env independent split
    elif mode=='environment':
        des_clean = pd.DataFrame(des_clean)
        val_train=(((des_clean['Enviroment']!='Environment3')))
        val_test =(((des_clean['Enviroment']=='Environment3')))
        des_train = des_clean[val_train].to_dict('records')
        des_test = des_clean[val_test].to_dict('records')
        assert len(des_clean)==(len(des_train)+len(des_test))
    elif mode=='gesture':
        des_clean = pd.DataFrame(des_clean)
        gestures = [f'Gesture{i}' for i in range(12)]
        val_test_indices = []
        for gesture in gestures:
            gesture_indices = des_clean[des_clean['Gesture'] == gesture].index.tolist()
            sampled_indices = random.sample(gesture_indices, min(10, len(gesture_indices)))
            val_test_indices.extend(sampled_indices)
        val_test = des_clean.index.isin(val_test_indices)
        val_train = ~val_test
        des_train = des_clean[val_train].to_dict('records')
        des_test = des_clean[val_test].to_dict('records')
        # des_test = des_clean.to_dict('records')
        # des_train = des_clean.to_dict('records')
        # assert len(des_clean)==(len(des_train)+len(des_test))
        print(des_clean[val_train]['Gesture'].value_counts())
        print(des_clean[val_test]['Gesture'].value_counts())

        
    return des_train, des_test

def save_result_confusion(y_ls, y_pred_ls, labels, name, path_des):
        """Save the confusion matrix information
        """
        cm = sklearn.metrics.confusion_matrix(y_ls, y_pred_ls, normalize='true')
        sns.set()
        fig, ax = plt.subplots()
        sns.heatmap(cm*100, annot=True, annot_kws={"size": 8}, fmt=".1f", linewidths=1., cmap='Greens', cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Target')
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title('Overall:'+str(round(((np.array(y_ls)==np.array(y_pred_ls)).mean())*100,1))+'%')
        os.makedirs(os.path.join(path_des), exist_ok=True)
        os.makedirs(os.path.join(path_des,'fig'), exist_ok=True)
        plt.savefig(path_des+'fig/cm-'+name+'.png', dpi=600, bbox_inches='tight')
        plt.close()

def save_result_statistics(y_ls, y_pred_ls, des, name, path_des):
    des = pd.DataFrame.from_dict(des)
    list_sub_all, list_env_all, list_ges_all, list_ord_all = list(des['Subject']), list(des['Enviroment']), list(des['Gesture']), list(des['Order'])
    list_sub_all = [int(sub.split('t')[-1]) for sub in list_sub_all]
    list_env_all = [int(env.split('t')[-1]) for env in list_env_all]
    list_ges_all = [int(ges.split('e')[-1]) for ges in list_ges_all]

    list_sub,list_env,list_ges,list_ord   = list(set(list_sub_all)), list(set(list_env_all)), list(set(list_ges_all)), list(set(list_ord_all))

    list_key = ['GS','GE','OS']   # gesture-subject, gesture-env, order-subject
    result_suc = {}
    result_all = {}
    result_suc['GS'] = pd.DataFrame(data=0, index=list_ges, columns=list_sub)
    result_suc['GE'] = pd.DataFrame(data=0, index=list_ges, columns=list_env)
    result_suc['OS'] = pd.DataFrame(data=0, index=list_ord, columns=list_sub)
    result_all['GS'] = pd.DataFrame(data=0, index=list_ges, columns=list_sub)  
    result_all['GE'] = pd.DataFrame(data=0, index=list_ges, columns=list_env)
    result_all['OS'] = pd.DataFrame(data=0, index=list_ord, columns=list_sub)
    suc_list = list(np.array(y_ls)==np.array(y_pred_ls))
    for i in range(len(des)):
        sub, env, ges, ord = list_sub_all[i], list_env_all[i], list_ges_all[i], list_ord_all[i]
        result_all['GS'].loc[ges,sub] += 1
        result_all['GE'].loc[ges,env] += 1
        result_all['OS'].loc[ord,sub] += 1
        if suc_list[i]:
            result_suc['GS'].loc[ges,sub] += 1
            result_suc['GE'].loc[ges,env] += 1
            result_suc['OS'].loc[ord,sub] += 1
    os.makedirs(os.path.join(path_des), exist_ok=True)
    os.makedirs(os.path.join(path_des,'fig'), exist_ok=True)
    sns.set()
    for key in list_key:
        fig, ax = plt.subplots()
        sns.heatmap((result_suc[key]/result_all[key])*100, annot=True, annot_kws={"size": 8}, fmt=".1f", linewidths=1., cmap='Greens', cbar=False)
        ax.set_title('Overall:'+str(round(((np.array(y_ls)==np.array(y_pred_ls)).mean())*100,1))+'%')
        plt.savefig(path_des+f'fig/{key}-'+name+'.png', dpi=600, bbox_inches='tight')
        plt.close()

def save_result_pred(result_pred, name, path_des):
    os.makedirs(os.path.join(path_des), exist_ok=True)
    os.makedirs(os.path.join(path_des,'result_pred'), exist_ok=True)
    with open(path_des+f'result_pred/'+name+'.pickle', 'wb') as f:
        pickle.dump(result_pred, f)