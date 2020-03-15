# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

import json
import datetime


INFO = {
    "description": "Domain Adaptation Iteration",
    "url": "",
    "version": "0.1.0",
    "year": 2020,
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
    parser = argparse.ArgumentParser(description="Inference on unlabeled nano store images")
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

    image_id = 1
    product_id = 1
    annotations_train = []

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
    out_dir = "/home/rui/photobox2real/generated_anno"

    imgs = os.listdir(img_dir)

    if len(imgs) == 0:
        print("COULD NOT FIND ANY IMAGE")
    else:
        print('Dataset len of: {}.'.format(len(imgs)))

    pbar = tqdm(total=len(imgs))
    for img in imgs:
        pbar.update(1)

        file_name = img
        img_cv2 = cv2.imread(os.path.join(img_dir, file_name))

        composite, predictions = coco_demo.run_on_opencv_image(img_cv2)
        product_boxes = predictions.bbox.numpy().tolist()

        if len(product_boxes) == 0:
            continue

        product_scores = predictions.extra_fields["scores"].numpy()
        product_labels = predictions.extra_fields["labels"].numpy()

        image_info = {'id': image_id, 'file_name': file_name, 'width': img_cv2.shape[1],
                      'height': img_cv2.shape[0], 'licence': 1, 'coco_url': ""}

        coco_output_train["images"].append(image_info)

        for idx, pred_bbox in enumerate(product_boxes):
            cls = int(product_labels[idx])
            if cls > 30:
                continue
            score = float(product_scores[idx])
            if score < 0.95:
                continue

            print('Good predictions: {}'.format(product_id))

            pred_bbox = [int(pos) for pos in pred_bbox]
            x1, y1, x2, y2 = pred_bbox
            w = x2 - x1
            h = y2 - y1

            annotation = {"image_id": image_id, "iscrowd": 0, "area": int(w * h),
                          "bbox": pred_bbox, "segmentation": [],
                          "id": product_id, "category_id": cls, "pred_score": score}

            annotations_train.append(annotation)
            product_id += 1
        # end-for
        image_id += 1
    # end-for
    pbar.close()
    coco_output_train["annotations"] = annotations_train

    with open('{}/annotations_photobox2real_iter1.json'.format(out_dir), 'w') as output_json_file:
        json.dump(coco_output_train, output_json_file)

# --config-file
# "models/all_data_skus200_v5/e2e_faster_rcnn_X_101_32x8d_FPN_1x_200skus.yaml"
# --img-dir
# "/home/ruichen/Documents/Documents_from_ubuntu_1604/Uncropped_data/cropped"
# --out-dir
# "/home/ruichen/Documents/Documents_from_ubuntu_1604/Uncropped_data/anno_prediction"


if __name__ == "__main__":
    main()
