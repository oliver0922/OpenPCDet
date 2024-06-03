from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .openset_center_head import Openset_CenterHead
from .openset_center_headv2 import Openset_CenterHeadV2
from .clip_test_center_head import Clip_test_CenterHead
from .voxelnext_head import VoxelNeXtHead
from .transfusion_head import TransFusionHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    "Openset_CenterHead" :Openset_CenterHead,
    "Openset_CenterHeadV2" :Openset_CenterHeadV2,
    "Clip_test_CenterHead" : Clip_test_CenterHead,
    'VoxelNeXtHead': VoxelNeXtHead,
    'TransFusionHead': TransFusionHead,
}
