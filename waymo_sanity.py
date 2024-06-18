import numpy as np
import pickle
from pathlib import Path

import math

import tensorflow as tf2
import tensorflow.compat.v1 as tf
from tensorflow import io
from tqdm import tqdm
import torch
import argparse

from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils as fu
import os

import cv2
from PIL import Image
import clip 

WAYMO_DATA_PATH = Path('/dataset/waymo')
DATA_PATH = Path('/dataset/waymo/waymo_processed_data_v0_5_0')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Waymo.')
    parser.add_argument('--run_device', type=int, help='multiframe_number', default=0)

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    
    gpu_id = args.run_device
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    
    with open('/dataset/waymo/waymo_infos_train_sampling_5.pkl', 'rb') as f:
        infos = pickle.load(f)

    current_tfrecord = ''
    #current_idx = 0
    
    for i, info in enumerate(tqdm(infos)):
        if i % 2 != 1:
            continue

        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        lidar_file = DATA_PATH / sequence_name / ('%04d.npy' % sample_idx)
        openseg_file = DATA_PATH / sequence_name / ('%04d_openseg.npz' % sample_idx)

        try:
            point_features = np.load(lidar_file) # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
        except: 
            print('LIDAR : ',lidar_file)

        try:
            point_features = np.load(openseg_file) # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
            feat = point_features['feat']
            mask_full = point_features['mask_full']
        except: 
            print('Openseg : ',openseg_file)
        
        