# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .cityscapes import Cityscapes
from .visual_pr import VisualPR
from .visual_pr_2cls import VisualPR_2cls
from .foggy_cityscapes import Foggy_Cityscapes
from .skus_box_online import SKUsBoxOnlineDataset

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "Cityscapes", "VisualPR", "Foggy_Cityscapes",
           "SKUsBoxOnlineDataset", "VisualPR_2cls"]
