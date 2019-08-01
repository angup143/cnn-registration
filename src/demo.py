from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from utils.utils import *
import os
import cv2
import math
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as tf
from skimage.transform import rotate

def transform_img(img, trans, rot, scale):
    rot_rad = rot*math.pi/180
    tform =  tf.SimilarityTransform(scale=scale,
            translation=trans, rotation=rot_rad)
    #to rotate around center
    shift_y, shift_x = np.array(img.shape[:2]) / 2.
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    tf_img = tf.warp(img,tf_shift+ (tform+tf_shift_inv), preserve_range=True)
    # tf_img = tf.warp(img,tform,preserve_range=True)
    return tf_img

# designate image path here
# IX_path = '../img/1a.jpg'
IX_path = '/home/ananya/Documents/rds-share/data/act_mapi/test/992_crop/2018/0.1.741_3.tif'
# IY_path = '../img/1b.jpg'
IY_path = '/home/ananya/Documents/rds-share/data/act_mapi/test/992_crop/2017/0.1.741_3_rot3.tif'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

tmp_x = np.arange(IX.shape[0])
tmp_y = np.arange(IX.shape[1])


tmp = np.swapaxes(np.asarray(np.meshgrid(tmp_x, tmp_y)),0,2)
tmp_image = np.ones(IX.shape).astype(np.uint16)
tmp_image[:,:,0:2] =tmp.astype(np.uint16)
#initialize
reg = Registration.CNN()
#register
X, Y, Z = reg.register(IX, IY)
#generate regsitered image using TPS
registered = tps_warp(Y, Z, IY, IX.shape)

registered_tmp =tps_warp(Y,Z,tmp_image, IX.shape)
registered_gt = transform_img(tmp_image,trans=(0,0),rot=-3,scale=1).astype(np.uint16)


# cb = checkboard(IX, registered_tmp, 11)

print(registered_tmp)
plt.subplot(131)
plt.title('registered_tmp')
plt.imshow(registered_tmp)
# plt.imshow(cv2.cvtColor(registered_tmp, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.title('registered')
plt.imshow(registered_gt)
# plt.imshow(cv2.cvtColor(registered_gt, cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.title('tmp')
# plt.imshow(cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB))
plt.imshow(tmp_image)
plt.show()

print(np.max(registered_tmp))
print(np.max(registered_gt))






    
