import os

import torch
import torchvision

from PIL import Image
from pycocotools.coco import COCO

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from .visual_pr import VisualPR

class MergeVisualPR(torchvision.datasets.coco.CocoDetection):

    def __init__(self, data_dir, split, is_training):

        #
        self._split = split
        self._data_dir = data_dir
        self._data_size = -1
        # self._data_handler = data_handler
        self._is_training = is_training

        self.ANCHORS = []
        # self.classes = []

        self.classes = ('__background__',  # always index 0
                        'ritz_medium',
                        'angies_boom_pop_chip',
                        'red_bull_red',
                        'ivory_concentrated_dishwashing',
                        'terra_chips',
                        'lays_potato_chips',
                        'dawn_ultra_dishwashing',
                        'equate_cotton_bandage',
                        'equate_exam_gloves',
                        'frosted_flakes',
                        'red_bull_sugar_free',
                        'nutter_butter_cookies',
                        'lysol_disinfecting',
                        'salted_cashew_halves',
                        'dawn_simply_clean',
                        'dawn_ultra_platinum',
                        'oreo_cookies',
                        'ritz_small',
                        'chips_ahoy',
                        'vita_coconut_water',
                        'red_bull_blue',
                        'bounty_napkins',
                        'ritz_large',
                        'red_bull_yellow',
                        'tostitos_scoops',
                        'veggie_straws',
                        'lays_stax_chips',
                        'tostitos_salsa',
                        'tide_detergent',
                        'equate_wound_dressing',)
        #
        all_dataset_name = os.listdir(data_dir)
        print (all_dataset_name, len(all_dataset_name))
        assert len(all_dataset_name) == 3, \
            'Error occurs when loading multiple datasets.'

        # self.len_0 = self.dataset_0.__len__()

        self.datasets_num = len(all_dataset_name)
        self.datasets = []
        self.datasets_len = []
        self.datasets_cum_len = []

        self.datasets = [visual_pr(data_dir=os.path.join(data_dir, dataset_name), split=split, is_training=is_training, dataset_name=dataset_name) \
                         for dataset_name in all_dataset_name]

        # self.datasets_len = [self.datasets.__len__() for i in range(len(all_dataset_name))]
        self.datasets_len = [dataset.__len__() for dataset in self.datasets]

        for i in range(len(all_dataset_name)):
            if i == 0:
                self.datasets_cum_len.append(self.datasets_len[0])
            else:
                self.datasets_cum_len.append(self.datasets_cum_len[i - 1] + self.datasets_len[i])

        print self.datasets_len

        print self.datasets_cum_len

    def __len__(self):
        # len_all_dataset = self.dataset_0.__len__() + \
        #                   self.dataset_1.__len__() + \
        #                   self.dataset_2.__len__() + \
        #                   self.dataset_3.__len__() + \
        #                   self.dataset_4.__len__()
        assert np.sum(self.datasets_len) == self.datasets_cum_len[-1], \
            'Error occurs when calculating the cum_sum.'

        return self.datasets_cum_len[-1]

    def __getitem__(self, idx):
        # if idx < self.dataset_0.__len__():
        #     return self.dataset_0.__getitem__(idx)
        # elif idx >= self.dataset_0.__len__():
        #     pass
        if idx < self.datasets_cum_len[0]:
            return self.datasets[0].__getitem__(idx)
        else:
            for i in range(1, self.datasets_num):
                if idx < self.datasets_cum_len[i]:
                    real_id = idx - self.datasets_cum_len[i - 1]
                    return self.datasets[i].__getitem__(real_id)


def get_loader(data_dir, split, is_training, batch_size=16, shuffle=True, num_workers=4):
    dataset = MergeVisualPR(data_dir, split, is_training)
    if is_training:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn_testing)


if __name__ == '__main__':

    # '/home/ruichen/Documents/2019_spring/data_p4'
    # '/home/ruichen/Documents/Documents_from_ubuntu_1604/p4_syn_data'

    datasets = MergeVisualPR(data_dir='/home/ruichen/Documents/Documents_from_ubuntu_1604/hand_crop_data', \
                             split='', is_training=True)
    print datasets.__len__()
