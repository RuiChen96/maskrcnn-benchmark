import os

import torch
import torchvision

from PIL import Image
from pycocotools.coco import COCO

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


"""
dir structure:
./data/
        data/citypersons
        data/citypersons/cityscape
            data/citypersons/cityscape/leftImg8bit/{train|val|test}
            data/citypersons/cityscape/gtFine/{train|val|test}
"""


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    # if _count_visible_keypoints(anno) >= min_keypoints_per_image:
    #     return True
    return False


class Foggy_Cityscapes(torchvision.datasets.coco.CocoDetection):

    def __init__(self, ann_file, root, beta=0.02, remove_images_without_annotations=False, transforms=None):
        # as you would do normally
        super(Foggy_Cityscapes, self).__init__(root, ann_file)

        # sort indices for reproducible results
        # self.ids is inheritance from __init__
        self.ids = sorted(self.ids)

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self._transforms = transforms

        self.beta = beta

    def __getitem__(self, idx):
        # load the image as a PIL Image
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        sub_folder = path.split('_')[0]
        # erfurt_000000_000019_leftImg8bit_foggy_beta_0.01.png
        path = path.split('.')[0] + '_foggy_beta_{}.png'.format(self.beta)
        img = Image.open(os.path.join(self.root, sub_folder, path)).convert('RGB')

        # img, anno = super(Cityscapes, self).__getitem__(idx)
        # anno = target

        # filter crowd annotations
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)   # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # and labels
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # create a BoxList from the boxes
        # boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        # boxlist.add_field("labels", labels)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # return the image, the boxlist and the idx in your dataset
        return img, target, idx

    def get_img_info(self, index):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk

        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]

        # return {"height": img_height, "width": img_width}
        return img_data


if __name__ == '__main__':
    data_dir = '/home/ruichen/data/CityPersons_DATASET/annotations_coco_format'
    img_dir = '/home/ruichen/Documents/Documents_from_ubuntu_1604/Foggy_Cityscapes_DATASET/leftImg8bit_foggy/train'
    annFile = os.path.join(data_dir, 'instancesonly_filtered_gtFine_train.json')
    # coco = COCO(annFile)
    # print(len(coco))
    data_loader = Foggy_Cityscapes(ann_file=annFile, root=img_dir, remove_images_without_annotations=False)
    img, target, idx = data_loader.__getitem__(0)
    img.show()
    print('target: {}. '.format(target))
    print('idx: {}. '.format(idx))
    print(data_loader.__len__())
