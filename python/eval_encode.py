import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from time import time
from scipy.io import savemat, loadmat

from fnc.encode import encode
from fnc.matching import matching

# --------------------------------------------------------------------------
#   Parameters
# --------------------------------------------------------------------------
DATA_PATH = 'E:/Dataset/CASIA-Iris-Lamp'
TEMP_DIR = './temp/'

# Feature encoding parameters
minWaveLength = 18
mult = 1
sigmaOnf = 0.5

# Matching parameters
thres = 0.38

# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    start = time()

    # Check the existence of temp_dir
    train_dir = os.path.join(TEMP_DIR, 'train/')
    test_dir = os.path.join(TEMP_DIR, 'test/')
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # read protocol
    train_name = []
    train_label = {}
    test_name = []
    test_label = {}
    with open(os.path.join(DATA_PATH, 'train.txt'), 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split(' ')
            train_name.append(tmp[0])
            train_label[tmp[0]] = tmp[1]
    with open(os.path.join(DATA_PATH, 'test.txt'), 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split(' ')
            test_name.append(tmp[0])
            test_label[tmp[0]] = tmp[1]

    end = time()
    print('\n>>> Loading time: {} [s]\n'.format(end - start))
    start = end

    # encoding
    for img_name in tqdm(train_name, ncols=75, ascii=True):
        iriscode = os.path.join(train_dir, "{}.mat".format(
            os.path.basename(img_name)))
        if not os.path.exists(iriscode):
            img = cv.imread(os.path.join(DATA_PATH, 'NormIm', img_name), 0)
            mask = np.zeros_like(img).astype(np.bool)
            template, mask = encode(img, mask, minWaveLength, mult, sigmaOnf)
            savemat(iriscode, mdict={'template': template, 'mask': mask})

    end = time()
    print('\n>>> Encoding time: {} [s]\n'.format(end - start))
    start = end

    # Matching
    currect_count = 0
    for img_name in tqdm(test_name, ncols=75, ascii=True):

        iriscode = os.path.join(test_dir, "{}.mat".format(
            os.path.basename(img_name)))
        cache = os.path.join(test_dir, "cache_{}.pkl".format(
            os.path.basename(img_name)))

        if os.path.exists(iriscode):
            matdata = loadmat(iriscode)
            template = matdata['template']
            mask = matdata['mask']
        else:
            img = cv.imread(os.path.join(DATA_PATH, 'NormIm', img_name), 0)
            mask = np.zeros_like(img).astype(np.bool)
            template, mask = encode(img, mask, minWaveLength, mult, sigmaOnf)
            savemat(iriscode, mdict={'template': template, 'mask': mask})
        result = matching(template, mask, train_dir, thres, cache)

        if isinstance(result, list):
            res = os.path.splitext(result[0])[0]
            if test_label[img_name] == train_label[res]:
                currect_count += 1
    print('Accuracy: {} %'.format(currect_count / len(test_name) * 100))

    end = time()
    print('\n>>> Eval time: {} [s]\n'.format(end - start))
