#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
import numpy as np
import cv2
from natsort import natsorted
import random
import sys

ROOT_DIR = '/media/juan/Data/retail_products/ceiling/shoot1_one_person_vivoteks'
IMAGE_DIR = os.path.join(ROOT_DIR, "all_cams")
VGG_ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "vgg_annotations")
SPLIT_TRAIN_TEST = False

INFO = {
    "description": "P5-real Dataset 2",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Juan Terven",
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
    no_regions_count = 0
    annotations_train = []
    annotations_test = []
    missing_annotations = []
    
    # Get list of images ending with ".png"
    image_names = []
    images_processed = []
    for image_name in os.listdir(IMAGE_DIR):
        if image_name.endswith(".png"):
            image_names.append(image_name)
    print(len(image_names), image_names)

    # Get list of files ending with ".json"
    vgg_jsons = []
    for json_file in os.listdir(VGG_ANNOTATIONS_DIR):
        if json_file.endswith(".json"):
            vgg_jsons.append(json_file)
    vgg_jsons = natsorted(vgg_jsons)
    print(vgg_jsons)

    if SPLIT_TRAIN_TEST:
        indices = list(range(0, len(image_names)))
        random.seed(a=1, version=2)
        training_indices = random.sample(range(1, len(image_names)), 4000)
        testing_indices = list(set(indices) - set(training_indices))
    else:
        training_indices = list(range(0, len(image_names)))
        testing_indices = []

    # go through each vgg json file
    for vgg_json in vgg_jsons:
        print(vgg_json)
        vgg_json_path = os.path.join(VGG_ANNOTATIONS_DIR, vgg_json)
        
        with open(vgg_json_path) as json_file:
            data = json.load(json_file)

        keys = list(data['_via_img_metadata'].keys())
        print('num keys:', len(keys))

        for key in keys:
            image_name = data['_via_img_metadata'][key]['filename']

            # search image file 
            if image_name in image_names and not(image_name in images_processed):
                image_filename = os.path.join(IMAGE_DIR, image_name)
                image = cv2.imread(image_filename)

                regions = data['_via_img_metadata'][key]['regions']
                if len(regions) > 0:
                    # save image info
                    image_info = {'id': image_id, 'file_name': image_name, 'width':image.shape[1],
                        'height': image.shape[0], 'licence': 1, 'coco_url': "" }

                    img_idx = image_names.index(image_name)
                    if img_idx in training_indices:
                        coco_output_train["images"].append(image_info)
                        print('Training:', image_name)
                    elif img_idx in testing_indices:
                        coco_output_test["images"].append(image_info)
                        print('Testing:', image_name)

                    # get annotations
                    regions = data['_via_img_metadata'][key]['regions']
                    for region in regions:
                        print(region)
                        if 'Class' in region['region_attributes']:
                            class_name = region['region_attributes']['Class']
                            x = region['shape_attributes']['x']
                            y = region['shape_attributes']['y']
                            w = region['shape_attributes']['width']
                            h = region['shape_attributes']['height']

                            cat_id = 0
                            for d in CATEGORIES:
                                if d["name"] == class_name:
                                    cat_id = d["id"]

                            if cat_id != 0:
                                annotation = {"image_id": image_id, "iscrowd": 0, "area": int(w*h),
                                            "bbox": [x, y, x+w, y+h], "segmentation": [],
                                            "id": product_id, "category_id": cat_id}

                                if img_idx in training_indices:
                                    annotations_train.append(annotation)
                                elif img_idx in testing_indices:
                                    annotations_test.append(annotation)
                                product_id += 1
                            else:
                                print("CATEGORY NOT FOUND:", class_name)
                        else:
                            missing_annotations.append(image_name)

                    image_id += 1
                    images_processed.append(image_name)
                else:
                    no_regions_count +=1

    coco_output_train["annotations"] = annotations_train
    coco_output_test["annotations"] = annotations_test

    with open('{}/annotations_train_cam29_crop.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output_train, output_json_file)
    if SPLIT_TRAIN_TEST:
        with open('{}/annotations_test_cam29_crop.json'.format(ROOT_DIR), 'w') as output_json_file:
            json.dump(coco_output_test, output_json_file)

    print(missing_annotations) 
    print('image_id:', image_id)
    print('no regions count:', no_regions_count)

if __name__ == "__main__":
    main()
