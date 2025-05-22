import numpy as np
import cv2
import matplotlib.pyplot as plt

cam_data = np.load('/home/shuboy/Desktop/Gesture_data/paper_data/20231206135332-0-cam.npy')
print(f"cam_data shape: {cam_data.shape}")

uD_data = np.load('/home/shuboy/Desktop/Gesture_data/paper_data/20231206135332-0-rad-uD.npy')
print(f"uD_data shape: {uD_data.shape}")
# Plot and save uD_data
plt.figure(figsize=(10, 6))
plt.imshow(uD_data, aspect='auto', cmap='jet', vmin=0, vmax=40)
plt.title('uD_data Frame 40')
plt.colorbar(label='Amplitude')
plt.xlabel('Range')
plt.ylabel('Doppler')
plt.tight_layout()
plt.savefig('/home/shuboy/Desktop/mmegogesture-fusion/uD_data.png')
plt.close()

# # Visualizating image
# # Normalize and convert to uint8 for display
# img = cam_data[40,...]
# if img.dtype != np.uint8:
#     img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#     img = img.astype(np.uint8)

# # cv2.imshow('Camera Data', img)
# cv2.imwrite('/home/shuboy/Desktop/mmegogesture-fusion/cam_data_cv2.png', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print(f"cam_data shape: {cam_data.shape}")
# print(f"cam_data mean: {np.mean(cam_data)}, std: {np.std(cam_data)}")
# radar_data = np.load('/home/shuboy/Desktop/Gesture_data/paper_data/20231206135332-0-rad-uD.npy')
# print(f"radar_data shape: {radar_data.shape}")
# print(f"radar_data mean: {np.mean(radar_data)}, std: {np.std(radar_data)}")