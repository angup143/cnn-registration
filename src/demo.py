from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from utils.utils import *
import os
import scipy
import cv2
import math
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as tf
from skimage.transform import rotate


def convert_coords_list_to_grid(est_coords, actual_coords,img_shape):

    est_combined_coords = est_coords[0] * img_shape[0] + est_coords[1] 
    actual_combined_coords =  actual_coords[0] * img_shape[0] + actual_coords[1] 
    ind = np.argsort(est_combined_coords)
    
    est_sorted = est_combined_coords[ind]
    actual_sorted = actual_combined_coords[ind]

    est_grid = np.zeros(img_shape)
    actual_grid = np.zeros(img_shape)

    est_grid[est_sorted//img_shape[0],est_sorted%img_shape[0]]



def transform_img_with_coords(img, trans, rot, scale):
    rot_rad = rot*math.pi/180
    img_shape = np.shape(img)

    tform =  tf.SimilarityTransform(scale=scale,
            translation=trans, rotation=rot_rad)
    #to rotate around center
    shift_y, shift_x = np.array(img.shape[:2]) / 2.
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    tf_img = tf.warp(img,tf_shift+ (tform+tf_shift_inv), preserve_range=True)

    warped_coords = tf.warp_coords(tf_shift + (tform + tf_shift_inv), img_shape)

    # tf_img = tf.warp(img,tform,preserve_range=True)
    return tf_img,warped_coords

# designate image path here
# IX_path = '../img/1a.jpg'
IX_path = '/home/ananya/Documents/rds-share/data/act_mapi/test/992_crop/2018/0.1.741_3.tif'
# IY_path = '../img/1b.jpg'
IY_path = '/home/ananya/Documents/rds-share/data/act_mapi/test/992_crop/2017/0.1.741_3_rot10.tif'

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
registered, est_coords, in_coords = tps_warp(Y, Z, IY, IX.shape)

# registered_tmp,_, _ = tps_warp(Y,Z,tmp_image, IX.shape)
registered_gt,warped_actual_coords = transform_img_with_coords(tmp_image,trans=(0,0),rot=-10,scale=1)

warped_estimated_coords = np.zeros([3,992,992])
warped_estimated_coords[0:2, est_coords[0],est_coords[1]]=np.asarray(in_coords)
warped_estimated_coords = np.repeat(np.expand_dims(warped_estimated_coords, axis=-1),3,axis=-1)
warped_estimated_coords[2] = [0,1,2]
# registered_tmp_2 = scipy.ndimage.map_coordinates(tmp_image, warped_estimated_coords)

registered_gt = registered_gt.astype(np.uint16)

rmse = np.sqrt(np.mean((warped_actual_coords - warped_estimated_coords)**2))
mae = np.mean(np.abs(warped_actual_coords - warped_estimated_coords))

print(rmse,mae)

cb = checkboard(IX, registered, 11)

# print(registered_tmp)
plt.subplot(131)
plt.title('IX')
plt.imshow(IX)
# plt.imshow(cv2.cvtColor(registered_tmp, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.title('registered')
plt.imshow(registered)
# plt.imshow(cv2.cvtColor(registered_gt, cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.title('cb')
# plt.imshow(cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB))
plt.imshow(cb)
plt.show()

print(np.max(registered_tmp))
print(np.max(registered_gt))






    
