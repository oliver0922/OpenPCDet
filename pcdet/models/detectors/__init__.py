from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .openset_centerpoint import Openset_CenterPoint
from .openset_kitti_train_centerpoint import Openset_Kitti_CenterPoint
from .openset_waymo_train_centerpoint import Openset_Waymo_CenterPoint
from .openset_centerpointv2 import Openset_CenterPointV2
from .openset_centerpointv3 import Openset_CenterPointV3
from .openset_centerpoint_test import Openset_CenterPoint_test
from .openset_centerpoint_recon_clip import Openset_CenterPoint_Recon_clip
from .openset_centerpoint_recon_clip_bev import Openset_CenterPoint_Recon_clip_BEV
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .transfusion import TransFusion
from .bevfusion import BevFusion

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'Openset_CenterPoint':Openset_CenterPoint,
    "Openset_Kitti_CenterPoint":Openset_Kitti_CenterPoint,
    "Openset_Waymo_CenterPoint":Openset_Waymo_CenterPoint,
    'Openset_CenterPointV2':Openset_CenterPointV2,
    "Openset_CenterPointV3":Openset_CenterPointV3,
    'Openset_CenterPoint_test':Openset_CenterPoint_test,
    "Openset_CenterPoint_Recon_clip":Openset_CenterPoint_Recon_clip,
    "Openset_CenterPoint_Recon_clip_BEV":Openset_CenterPoint_Recon_clip_BEV,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet,
    'VoxelNeXt': VoxelNeXt,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
