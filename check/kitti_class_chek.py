import pickle
import numpy as np
import tqdm

with open('/home/OpenPCDet/data/kitti/kitti_infos_trainval.pkl', 'rb') as f:
	full = pickle.load(f)
 

for i in range(len(full)):
    print(i)
    part = full[i]
    part_anno = part['annos']
    
    name = part_anno['name']
    if 'Van' in name:
        print("1")