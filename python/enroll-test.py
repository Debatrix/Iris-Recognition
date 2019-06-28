import os
import cv2
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

from fnc.segment import segment
from fnc.normalize import normalize
from fnc.encode import encode

# -----------------------------------------------------------------------------
# Parameters for extracting feature
# (The following parameters are default for CASIA1 dataset)
# -----------------------------------------------------------------------------
im_filename = 'CASIA1/1/001_1_1.jpg'
TEMP_DIR = './temp/'

# Segmentation parameters
eyelashes_thres = 80

# Normalisation parameters
radial_res = 20
angular_res = 240

# Feature encoding parameters
minWaveLength = 18
mult = 1
sigmaOnf = 0.5

if __name__ == "__main__":

    im = cv2.imread(im_filename, 0)

    # [x,y,r], [x,y,r], float 0~nan~255
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, False)

    img1 = imwithnoise.copy()

    # float 0~1, bool
    polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0],
                                         ciriris[2], cirpupil[1], cirpupil[0],
                                         cirpupil[2], radial_res, angular_res)

    # float 0-1, float 0-1
    template, mask = encode(polar_array, noise_array, minWaveLength, mult,
                            sigmaOnf)

    print(ciriris, cirpupil)
    print(polar_array.shape)
    print(template.size)

    fig = plt.figure()

    plt.subplot(321)
    img1[np.isnan(img1)] = 0
    plt.imshow(img1, cmap='gray')

    plt.subplot(322)
    im = cv2.circle(im, (ciriris[1], ciriris[0]), ciriris[2], (255, 0, 255), 1)
    im = cv2.circle(im, (cirpupil[1], cirpupil[0]), cirpupil[2], (0, 255, 255),
                    1)
    plt.imshow(im)

    plt.subplot(323)
    polar_array[np.isnan(polar_array)] = 0
    plt.imshow(polar_array, cmap='gray')

    plt.subplot(324)
    noise_array[np.isnan(noise_array)] = 0
    plt.imshow(noise_array.astype(np.float64), cmap='gray')

    plt.subplot(325)
    template[np.isnan(template)] = 0
    plt.imshow(template, cmap='gray')

    plt.subplot(326)
    mask[np.isnan(mask)] = 0
    plt.imshow(mask, cmap='gray')

    plt.show()
