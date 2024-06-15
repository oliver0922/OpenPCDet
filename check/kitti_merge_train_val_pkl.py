import pickle


with open('/home/OpenPCDet/data/kitti/kitti_infos_train.pkl', 'rb') as f:
	train = pickle.load(f)
 
 
with open('/home/OpenPCDet/data/kitti/kitti_infos_val.pkl', 'rb') as f:
	val = pickle.load(f)
 
 
with open('/home/OpenPCDet/data/kitti/kitti_infos_trainval.pkl', 'rb') as f:
	trainval = pickle.load(f)
 
 
print("1")