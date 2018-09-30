import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    data = []
    df_train = pd.read_csv('../input/test_v3.csv') # Memory hungry, use test data instead

    count = 0
    # for file in tqdm(df_train['image_name'], miniters=256):
    for file in df_train['image_name']:
        img = cv2.imread('../input/test-jpg/{}.jpg'.format(file))
        data.append(img)
        count += 1
        if count > 10000: # Memory hungry, pick some images
            break

    data = np.array(data, np.float32) / 255. # Must use float32 at least otherwise we get over float16 limits
    print("Shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

# Result running on kaggle amazon training data
# 100%|█████████████████████████████████████████████| 40479/40479 [00:29<00:00, 1395.58it/s]
# Shape:  (40479, 256, 256, 3)
# means: [0.30275333, 0.34446728, 0.31535667]
# stdevs: [0.13733844, 0.14379855, 0.16717975]
# transforms.Normalize(mean = [0.30275333, 0.34446728, 0.31535667], std = [0.13733844, 0.14379855, 0.16717975])
