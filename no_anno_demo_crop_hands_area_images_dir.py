# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import torch
import numpy as np

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

# for COCO annotations
from pycocotools.coco import COCO

import time
import math
import json

import datetime
import re
import fnmatch
from natsort import natsorted
import random
import sys


ROOT_DIR = '/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/'
# IMAGE_DIR = os.path.join(ROOT_DIR, "all_cams")
# VGG_ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "vgg_annotations")
# SPLIT_TRAIN_TEST = False

INFO = {
    "description": "P5-Real Ceiling Camera View-29",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Rui Chen",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'ritz_medium',
        'supercategory': 'sku',
    },
    {
        'id': 2,
        'name': 'angies_boom_pop_chip',
        'supercategory': 'sku',
    },
    {
        'id': 3,
        'name': 'red_bull_red',
        'supercategory': 'sku',
    },
    {
        'id': 4,
        'name': 'ivory_concentrated_dishwashing',
        'supercategory': 'sku',
    },
    {
        'id': 5,
        'name': 'terra_chips',
        'supercategory': 'sku',
    },
    {
        'id': 6,
        'name': 'lays_potato_chips',
        'supercategory': 'sku',
    },
    {
        'id': 7,
        'name': 'dawn_ultra_dishwashing',
        'supercategory': 'sku',
    },
    {
        'id': 8,
        'name': 'equate_cotton_bandage',
        'supercategory': 'sku',
    },
    {
        'id': 9,
        'name': 'equate_exam_gloves',
        'supercategory': 'sku',
    },
    {
        'id': 10,
        'name': 'frosted_flakes',
        'supercategory': 'sku',
    },
    {
        'id': 11,
        'name': 'red_bull_sugar_free',
        'supercategory': 'sku',
    },
    {
        'id': 12,
        'name': 'nutter_butter_cookies',
        'supercategory': 'sku',
    },
    {
        'id': 13,
        'name': 'lysol_disinfecting',
        'supercategory': 'sku',
    },
    {
        'id': 14,
        'name': 'salted_cashew_halves',
        'supercategory': 'sku',
    },
    {
        'id': 15,
        'name': 'dawn_simply_clean',
        'supercategory': 'sku',
    },
    {
        'id': 16,
        'name': 'dawn_ultra_platinum',
        'supercategory': 'sku',
    },
    {
        'id': 17,
        'name': 'oreo_cookies',
        'supercategory': 'sku',
    },
    {
        'id': 18,
        'name': 'ritz_small',
        'supercategory': 'sku',
    },
    {
        'id': 19,
        'name': 'chips_ahoy',
        'supercategory': 'sku',
    },
    {
        'id': 20,
        'name': 'vita_coconut_water',
        'supercategory': 'sku',
    },
    {
        'id': 21,
        'name': 'red_bull_blue',
        'supercategory': 'sku',
    },
    {
        'id': 22,
        'name': 'bounty_napkins',
        'supercategory': 'sku',
    },
    {
        'id': 23,
        'name': 'ritz_large',
        'supercategory': 'sku',
    },
    {
        'id': 24,
        'name': 'red_bull_yellow',
        'supercategory': 'sku',
    },
    {
        'id': 25,
        'name': 'tostitos_scoops',
        'supercategory': 'sku',
    },
    {
        'id': 26,
        'name': 'veggie_straws',
        'supercategory': 'sku',
    },
    {
        'id': 27,
        'name': 'lays_stax_chips',
        'supercategory': 'sku',
    },
    {
        'id': 28,
        'name': 'tostitos_salsa',
        'supercategory': 'sku',
    },
    {
        'id': 29,
        'name': 'tide_detergent',
        'supercategory': 'sku',
    },
    {
        'id': 30,
        'name': 'equate_wound_dressing',
        'supercategory': 'sku',
    }
]


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--img-dir",
        default="",
        help="images directory",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="output images directory",
    )
    parser.add_argument(
        "--annotations-file",
        default="..annotations.json",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # init for coco annotations
    coco_output_train = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    coco_output_test = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    product_id = 1
    # no_regions_count = 0
    annotations_train = []
    # annotations_test = []
    # missing_annotations = []

    image_names = []
    # images_processed = []

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    # load images from directory
    img_dir = args.img_dir

    # load annotations
    # cam29, cam30, cam31, cam100
    camera_view = 'cam100'
    # file name: annotations_coco_cam29.json
    # annFile = os.path.join(img_dir, '{}.json'.format('annotations_coco_' + camera_view))
    annFile = os.path.join(img_dir, '{}.json'.format('annotations_train'))

    coco = COCO(annFile)

    imgs = []

    for img_id in coco.imgs:
        imgs.append((img_id, coco.imgs[img_id], 0))

    if len(imgs) == 0:
        print("COULD NOT FIND ANY IMAGE")
    else:
        print('dataset len of: {}.'.format(len(imgs)))

    out_dir = args.out_dir

    himg_count = 1
    for img in imgs:
        start_time = time.time()

        img_id = img[0]
        file_name = img[1]['file_name']
        img_cv2 = cv2.imread(os.path.join(img_dir, file_name))

        composite, predictions = coco_demo.run_on_opencv_image(img_cv2)

        # cv2.imwrite(os.path.join(out_dir, 'test_keypoint.jpg'), composite)

        print("Time: {:.2f} s / img".format(time.time() - start_time))

        composite = cv2.resize(composite, None, fx=0.5, fy=0.5)

        human_labels = predictions.get_field("labels").numpy().tolist()
        human_boxes = predictions.bbox.numpy().tolist()

        # get the keypoints
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()

        print('labels:', human_labels)
        print('boxes:', human_boxes)
        print('keypoints:', kps.shape, kps)

        # load annotations for each input img
        annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(annIds)
        classes = []
        bboxes = []
        for ann in anns:
            cat_id = ann['category_id']

            if ann['iscrowd']:
                cat_id = -1
            classes.append(cat_id)
            bboxes.append(ann['bbox'])
        classes = np.asarray(classes)
        bboxes = np.asarray(bboxes)
        classes = classes.astype(np.float32)
        bboxes = bboxes.astype(np.float32)

        if len(bboxes) == 0:
            continue

        # calculate bboxes center
        bboxes_x1 = bboxes[:, 0]
        bboxes_y1 = bboxes[:, 1]
        bboxes_x2 = bboxes[:, 2]
        bboxes_y2 = bboxes[:, 3]

        # bboxes_x_bar = (bboxes_x1 + bboxes_x2) / 2.0
        # bboxes_y_bar = (bboxes_y1 + bboxes_y2) / 2.0
        bboxes_x_bar = (bboxes[0, 0] + bboxes[0, 2]) / 2.0
        bboxes_y_bar = (bboxes[0, 1] + bboxes[0, 3]) / 2.0

        # bboxes_centers = np.hstack((bboxes_x_bar, bboxes_y_bar))
        bboxes_centers = (bboxes_x_bar, bboxes_y_bar)
        print('bboxes_centers: ', bboxes_centers)

        # crop the hands
        # if kps.shape[0] == 1:
        # pid means person_id
        for pid in range(kps.shape[0]):
            kps_names = PersonKeypoints.NAMES
            # h1 denotes left_hand, h2 denotes right_hand
            h1xy = kps[pid, kps_names.index('left_wrist'), :2]
            h2xy = kps[pid, kps_names.index('right_wrist'), :2]

            print('h1xy:', h1xy)
            print('h2xy:', h2xy)

            dist = math.sqrt((h1xy[0]-h2xy[0])**2 + (h1xy[1]-h2xy[1])**2)
            print('dist between 2 hands:', dist)

            dist_to_h1 = math.sqrt((bboxes_centers[0] - h1xy[0]) ** 2 + (bboxes_centers[1] - h1xy[1]) ** 2)
            dist_to_h2 = math.sqrt((bboxes_centers[0] - h2xy[0]) ** 2 + (bboxes_centers[1] - h2xy[1]) ** 2)
            print('dist for product to h1: ', dist_to_h1)
            print('dist for product to h2: ', dist_to_h2)

            hand_images = []
            # if the hands are separated, crop one image for each hand
            img_size = 800

            if dist_to_h1 > 500.0 and dist_to_h2 > 500.0:
                # wrong person (grab no products) is detected
                continue

            if dist > 400.0:
                if dist_to_h1 < dist_to_h2:
                    h1, position1 = crop_image(img_cv2, [h1xy[0], h1xy[1]], img_size)

                    new_bboxes = re_calculate_bboxes(bboxes, [h1xy[0], h1xy[1]], img_size, position1)

                    hand_images.append(h1)
                else:
                    h2, position2 = crop_image(img_cv2, [h2xy[0], h2xy[1]], img_size)

                    new_bboxes = re_calculate_bboxes(bboxes, [h2xy[0], h2xy[1]], img_size, position2)

                    hand_images.append(h2)
            else:
                # if both hands are close,
                # we only need to crop around the middle of them
                hmiddle = [(h1xy[0] + h2xy[0])/2, (h1xy[1] + h2xy[1])/2]
                h1, position = crop_image(img_cv2, hmiddle, img_size)

                new_bboxes = re_calculate_bboxes(bboxes, hmiddle, img_size, position)

                hand_images.append(h1)

            # save the hand images
            for idx, hand_image in enumerate(hand_images):
                print(new_bboxes)

                # visualization
                # for i, bbox in enumerate(new_bboxes):
                #     cv2.rectangle(hand_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 0, 1)
                #     cv2.putText(hand_image, str(classes[i]), (bbox[0], bbox[1] - 8), 1, 1, 2, 1)
                # visualization

                cv2.imshow("hand " + str(idx + 1), hand_image)
                cv2.imwrite(os.path.join(out_dir, str(himg_count)+".jpg"), hand_image)

                image_name = str(himg_count) + ".jpg"
                image_names.append(image_name)

                current_img = cv2.imread(os.path.join(out_dir, image_name))

                # insert my code here

                image_info = {'id': image_id, 'file_name': image_name, 'width': current_img.shape[1],
                              'height': current_img.shape[0], 'licence': 1, 'coco_url': ""}

                coco_output_train["images"].append(image_info)
                print('Training:', image_name)

                for i, bbox in enumerate(new_bboxes):
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1

                    cls = int(classes[i])

                    annotation = {"image_id": image_id, "iscrowd": 0, "area": int(w * h),
                                  "bbox": bbox, "segmentation": [],
                                  "id": product_id, "category_id": cls}

                    annotations_train.append(annotation)
                    product_id += 1

                # insert my code end here

                himg_count += 1
                image_id += 1

        cv2.imshow("Detections", composite)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

    # save coco annotations
    coco_output_train["annotations"] = annotations_train

    with open('{}/annotations_test_{}_crop_{}_batch3.json'.format(ROOT_DIR, camera_view, img_size), 'w') as output_json_file:
        json.dump(coco_output_train, output_json_file)


def crop_image(image, center, crop_size):
    imgx1 = int(center[0]-crop_size/2)
    imgx2 = int(center[0]+crop_size/2)
    imgy1 = int(center[1]-crop_size/2)
    imgy2 = int(center[1]+crop_size/2)

    print('image shape:', image.shape)
    print('x1, x2, y1, y2:', imgx1, imgx2, imgy1, imgy2)

    offx1 = 0
    offx2 = 0
    offy1 = 0
    offy2 = 0
    if imgx1 < 0:
        imgx1 = 0
        offx2 = -(imgx1)
    if imgx2 >= image.shape[1]:
        offx1 =  imgx2 - image.shape[1] + 1
        imgx2 = image.shape[1]-1
        

    if imgy1 < 0:
        imgy1 = 0
        offy2 = -(imgy1)
    if imgy2 >= image.shape[0]:
        offy1 =  imgy2 - image.shape[0] + 1
        imgy2 = image.shape[0]-1
        
    
    print('offx1, offx2, offy1, offy2:', offx1, offx2, offy1, offy2)
    img_c = image[imgy1-offy1:imgy2+offy2, imgx1-offx1:imgx2+offx2]

    # img + (x1, y1, x2, y2)
    return img_c, (imgx1-offx1, imgy1-offy1, imgx2+offx2, imgy2+offy2)


def re_calculate_bboxes(bboxes, hand_center, crop_size, img_patch):
    new_bboxes = []

    # top_left = [int(hand_center[0] - (crop_size / 2.0)), int(hand_center[1] - (crop_size / 2.0))]
    # bottom_right = [int(hand_center[0] + (crop_size / 2.0)), int(hand_center[1] + (crop_size / 2.0))]

    top_left = [img_patch[0], img_patch[1]]
    bottom_right = [img_patch[2], img_patch[3]]

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # x1 -= top_left[0]
        # y1 -= top_left[1]
        # x2 -= top_left[0]
        # y2 -= top_left[1]
        #
        # x1 = int(x1)
        # y1 = int(y1)
        # x2 = int(x2)
        # y2 = int(y2)

        if x1 < top_left[0]:
            x1 = 0
            # x1 = top_left[0]
        else:
            x1 -= (top_left[0] + 1)

        if y1 < top_left[1]:
            y1 = 0
            # y1 = top_left[1]
        else:
            y1 -= (top_left[1] + 1)

        if x2 > bottom_right[0]:
            # x2 = bottom_right[0]
            x2 = bottom_right[0] - (top_left[0] + 1)
        else:
            x2 -= (top_left[0] + 1)

        if y2 > bottom_right[1]:
            y2 = bottom_right[1] - (top_left[1] + 1)
        else:
            y2 -= (top_left[1] + 1)

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        print(x1, y1, x2, y2)

        new_bboxes.append([x1, y1, x2, y2])

    return new_bboxes


if __name__ == "__main__":
    main()
