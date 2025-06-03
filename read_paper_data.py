import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

folder = '/bigdata/shuboy/mm-egogesture/preprocessed_data/in_distribution_gestures/May21/resized_data/'
# '/bigdata/shuboy/mm-egogesture/Gesture_processed_public/'

# file_name = '20231207184556-5'
# file_name_cam = file_name+ '-img'
file_name_cam = "20250521120816-seg-0-Class0-img"
img = np.load(folder + file_name_cam + '.npy')
print(f"cam_data shape: {img.shape}")
# img = np.transpose(img, (1, 2, 0))
print(f"cam_data shape: {img.shape}")
if img.dtype != np.uint8:
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(file_name_cam+'.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

file_name_uD = file_name + '-rad-uD'  #20231207160300-1-rad-uD
uD_data = np.load(folder + file_name_uD + '.npy')
print(f"uD_data shape: {uD_data.shape}")
# Plot and save uD_data
plt.figure(figsize=(10, 6))
plt.imshow(uD_data, aspect='auto', cmap='jet', vmin=0, vmax=40)
plt.colorbar(label='Amplitude')
plt.xlabel('Range')
plt.ylabel('Doppler')
plt.tight_layout()
plt.savefig(file_name_uD+'.png')
plt.close()


# cam_mean = []
# cam_std = []
# rad_mean = []
# rad_std = []
# for cam_file in glob.glob(folder + '*-img.npy'): 
#     cam_data = np.load(cam_file)
#     channel_mean = np.mean(cam_data, axis=(1,2))
#     channel_std = np.std(cam_data, axis=(1,2))
#     cam_mean.append(channel_mean)
#     cam_std.append(channel_std)

# for rad_file in glob.glob(folder + '*-rad-uD.npy'):
#     rad_data = np.load(rad_file)
#     if np.mean(rad_data)<20:
#         rad_mean.append(np.mean(rad_data))
#     if np.std(rad_data)<20:
#         rad_std.append(np.std(rad_data))
# print('cam mean', np.mean(cam_mean, axis=0))
# print('cam std', np.mean(cam_std, axis=0))
# print('rad mean', np.mean(rad_mean))
# print('rad std', np.mean(rad_std))