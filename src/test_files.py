import os
import cv2
import math
import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as tf
from skimage.transform import rotate
from skimage.transform import AffineTransform, FundamentalMatrixTransform, SimilarityTransform, warp

import Registration
from utils.utils import *

def get_warped_coords(img, rot):
    rot_rad = rot*math.pi/180
    img_shape = np.shape(img)

    tform =  tf.SimilarityTransform(rotation=rot_rad)
    #to rotate around center
    shift_y, shift_x = np.array(img.shape[:2]) / 2.
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    warped_coords = tf.warp_coords(tf_shift + (tform + tf_shift_inv), img_shape)



if __name__ == '__main__':
    csv_file = '/home/ananya/rds-share/data/act_mapi/test/urban_test_files/test.txt'
    output_txtfile = '/home/ananya/rds-share/data/act_mapi/test/urban_test_files/2017_cnnreg_results.txt'
    
    metrics_rmse = []
    metrics_mae = []
    output_rows = []
    reg = Registration.CNN()


    with open(csv_file) as details_file:
        csv_reader = csv.reader(details_file)
        for row in csv_reader:
            reference_img_path = re.sub('2017', '2018', row[0])
            print('reference img', reference_img_path)
            query_img_path = row[1]
            rot = row[2]
            
            query_img = cv2.imread(query_img_path, 1)
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            reference_img = cv2.imread(reference_img_path, 1)
            reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

            X, Y, Z = reg.register(reference_img, query_img)
            registered_img, est_coords, in_coords = tps_warp(Y, Z, query_img, reference_img.shape)

            warped_estimated_coords = np.zeros(reference_img.shape)
            warped_estimated_coords[0:2, est_coords[0],est_coords[1]]=np.asarray(in_coords)
            warped_estimated_coords = np.repeat(np.expand_dims(warped_estimated_coords, axis=-1),3,axis=-1)
            warped_estimated_coords[2] = [0,1,2]

            warped_actual_coords = get_warped_coords(reference_img, rot)

            rmse = np.sqrt(np.mean((warped_actual_coords - warped_estimated_coords)**2))
            mae = np.mean(np.abs(warped_actual_coords - warped_estimated_coords))

            print('Image: {}, rmse:{}, mae:{}, gt_rot:{}'.format(query_img_path,rmse,mae,rot))
            output_rows.append(query_img_path,rmse,mae,rot)

            metrics_mae.append(mae)
            metrics_rmse.append(rmse)

    with open(output_txtfile,'w') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(output_rows)

    

                    
