"""
    Dataset loader for Home-baked retail checkout dataset
    It only uses bounding boxes so it can only be used for FasterRCNN

    Juan Terven, March 2019
    AiFi Inc.
"""
import os
import cv2
import numpy as np
import json
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from random import randint
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torchvision.transforms import functional as F
import torch.multiprocessing
from .src_lib_datasets_unlited_data_generation import unlimited_data_generator

torch.multiprocessing.set_sharing_strategy('file_system')

class SKUsBoxOnlineDataset(torch.utils.data.Dataset):

    def __init__(self, skus_dir, bgs_dir, hands_dir, training, transforms=None):

        self.transforms = transforms
        product_dir = skus_dir
        bg_dir = bgs_dir
        hand_dir = hands_dir
        self.img = np.empty((400, 400, 3))
        self.online_generator = unlimited_data_generator(product_dir,
                                                         bg_dir,
                                                         hand_dir,
                                                         None)

    def __getitem__(self, idx):

        annotation = self.online_generator.generate()
        augmented = self.online_generator.augmentation(annotation)

        self.img = augmented['image'].copy()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h, w, _ = self.img.shape
        boxes = []
        classes = []
        for idx, bbox in enumerate(augmented['bboxes']):
            boxes.append(bbox)
            classes.append(augmented['category_id'][idx])

        #print('boxes:', type(boxes), boxes)
        #print('classes:', type(classes), classes)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = BoxList(boxes, (w, h), mode="xywh").convert("xyxy")
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        #target = target.clip_to_image(remove_empty=True)

        # Convert numpy array to PIL Image
        img = Image.fromarray(self.img)
        img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return 1000000

    def get_img_info(self, index):
        im_info = tuple(map(int, (str(self.img.shape[0]), str(self.img.shape[1]))))
        return {"height": im_info[0], "width": im_info[1]}

