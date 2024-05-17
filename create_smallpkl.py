import pickle

with open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train_with_clip_agnostic.pkl', 'rb') as f:
	full = pickle.load(f)
# print(full)

partial = full[:1]
pickle.dump(partial, open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train_with_clip_agnostic_overfit_1.pkl', 'wb'))
