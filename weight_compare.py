import torch

# Weight 파일 경로
weight_path1 = '/home/OpenPCDet/output/cfgs/openset/centerpoint-nuscenes_detection_freeze_clip_train/default/ckpt/checkpoint_epoch_523.pth'
weight_path2 = '/home/OpenPCDet/output/combined_weight_detection30_reconlatest.pth'

# Weight 파일 로드
weights1 = torch.load(weight_path1, map_location=torch.device('cpu'))
weights2 = torch.load(weight_path2, map_location=torch.device('cpu'))

# Layer 이름 추출
layers1 = set(weights1['model_state'].keys())
layers2 = set(weights2['model_state'].keys())

# 겹치는 레이어와 겹치지 않는 레이어 찾기
common_layers = layers1.intersection(layers2)
unique_to_weight1 = layers1 - layers2
unique_to_weight2 = layers2 - layers1

# 겹치는 레이어의 가중치 비교
matching_layers = []
non_matching_layers = []

for layer in common_layers:
    if torch.equal(weights1['model_state'][layer], weights2['model_state'][layer]):
        matching_layers.append(layer)
    else:
        non_matching_layers.append(layer)

# 결과 출력
print(f"Matching layers: {matching_layers}")
print(f"Non-matching layers: {non_matching_layers}")
print(f"Layers unique to weight file 1: {unique_to_weight1}")
print(f"Layers unique to weight file 2: {unique_to_weight2}")