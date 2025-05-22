import numpy as np

data = np.load('/bigdata/shuboy/mm-egogesture/Gesture_processed_public/20231206135332-0-rad-uD.npy')
# camera (80, 256, 256, 3)
# img: (3, 256, 256)
# (128, 512)
print(data.shape)