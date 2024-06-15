import numpy as np
import pickle
from pathlib import Path
from waymo_open_dataset import dataset_pb2


DATA_PATH = Path('/home/OpenPCDet/data/waymo/waymo_processed_data_v0_5_0')


if __name__ == '__main__':
    
    with open('/home/OpenPCDet/data/waymo/waymo_infos_train_sampling_5.pkl', 'rb') as f:
        infos = pickle.load(f)
    
    for info in infos:
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        lidar_file = DATA_PATH / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file) # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]