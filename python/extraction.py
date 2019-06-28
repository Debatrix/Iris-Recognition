import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from time import time
import pickle
from itertools import combinations, repeat
from sklearn.preprocessing import LabelBinarizer
from multiprocessing import Pool, cpu_count

from fnc.encode import encode

# --------------------------------------------------------------------------
#   Parameters
# --------------------------------------------------------------------------
DATA_PATH = '/home/dl/wangleyuan/dataset/CASIA-Iris-Thousand'
LESS_FLAG = False
# DATA_PATH = 'E:/Dataset/CASIA-Iris-Lamp'
# LESS_FLAG = True
TEMP_DIR = './feature/'

# Feature encoding parameters
minWaveLength = 18
mult = 1
sigmaOnf = 0.5


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def calHammingDist(template1, template2):
    # Initialize
    hd = np.nan

    # Shift template left and right, use the lowest Hamming distance
    for shifts in range(-8, 9):
        template1s = shiftbits(template1, shifts)

        totalbits = template1s.size

        hd1 = np.logical_xor(template1s, template2).sum() / template1s.size

        if hd1 < hd or np.isnan(hd):
            hd = hd1

    return hd


def shiftbits(template, noshifts):
    # Initialize
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    # Shift
    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew


def getHmdistMat(features):
    num_feature = features.shape[0]
    hm_dists = np.zeros((num_feature, num_feature))
    pairs = [x for x in combinations([y for y in range(num_feature)], 2)]
    for x, y in tqdm(pairs, ncols=75, ascii=True):
        hm_dists[x, y] = calHammingDist(features[x, :, :], features[y, :, :])

    hm_dists = hm_dists + hm_dists.T

    return hm_dists


def getfeaturs(img_names, mask, size=(70, 1080)):
    features = np.zeros((len(img_names), size[0], size[1]), dtype=np.bool)
    for idx in tqdm(range(len(img_names)), ncols=75, ascii=True):
        img = cv.imread(os.path.join(DATA_PATH, 'NormIm', img_names[idx]), 0)
        features[idx, :, :], _ = encode(img, mask, minWaveLength, mult,
                                        sigmaOnf)
    return features


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(DATA_PATH)
    # Check the existence of temp_dir
    ft_path = os.path.join(TEMP_DIR, DATA_PATH.split('/')[-1] + '_shift.pkl')

    # read protocol
    start = time()
    img_names = []
    labels = []
    with open(os.path.join(DATA_PATH, 'test.txt'), 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split(' ')
            img_names.append(tmp[0])
            labels.append(tmp[1])
    if LESS_FLAG:
        img_names = img_names[:15]
        labels = labels[:15]
    onehot = LabelBinarizer().fit_transform(labels)

    end = time()
    print('\n>>> Loading time: {} [s]\n'.format(end - start))
    start = end

    # encoding
    mask = np.zeros((70, 540)).astype(np.bool)
    features = getfeaturs(img_names, mask, (70, 1080))

    end = time()
    print('\n>>> Encoding time: {} [s]\n'.format(end - start))
    start = end

    # mask = np.zeros((70, 1080)).astype(np.bool)
    hm_dists = getHmdistMat(features)

    end = time()
    print('\n>>> calHammingDist time: {} [s]\n'.format(end - start))

    ft_load = {
        'onehot': onehot,
        'features': features.reshape(len(img_names), -1),
        'labels': labels,
        'hm_dists': hm_dists
    }
    if not LESS_FLAG:
        with open(ft_path, 'wb') as f:
            pickle.dump(ft_load, f)
