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

import glob

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
    
    

    
    tfrecord_files = sorted(glob.glob(os.path.join(WAYMO_DATA_PATH,'training','*.tfrecord')))

    for tf_record_path in tqdm(tfrecord_files):
        poses = []
        dataset = tf.data.TFRecordDataset(tf_record_path, compression_type='')

        sequence_name = tf_record_path.split('/')[-1].split('.')[0]
        pose_file = DATA_PATH / sequence_name / 'poses.npy'
        pt_num = len(glob.glob(os.path.join(DATA_PATH,sequence_name,'0*.npy')))

        
        cnt = 0
        for raw_record in dataset:
            data = raw_record
            frame = dataset_pb2.Frame()
            frame.ParseFromString(data.numpy())
            frame_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
            poses.append(frame_pose)
            import pdb; pdb.set_trace()


            cnt+=1
        poses = np.stack(poses, axis=0)
        np.save(pose_file, poses)

        
    '''
    for i, info in enumerate(tqdm(infos)):

        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        lidar_file = DATA_PATH / sequence_name / ('%04d.npy' % sample_idx)

        tf_record_path = WAYMO_DATA_PATH / 'training' / f'{sequence_name}.tfrecord'
        pose_file = DATA_PATH / sequence_name / 'poses.npy'
        
        if tf_record_path != current_tfrecord:
            if cnt > 0:
                poses = np.stack(poses, axis=0)
                np.save(pose_file, poses)
                poses = []
            current_tfrecord = tf_record_path
            dataset = tf.data.TFRecordDataset(tf_record_path, compression_type='')

        for raw_record in dataset.skip(sample_idx).take(1):
            data = raw_record
        
        frame = dataset_pb2.Frame()
        frame.ParseFromString(data.numpy())

        frame_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
        poses.append(frame_pose)

        cnt += 1

    poses = np.stack(poses, axis=0)
    np.save(pose_file, poses)
    '''