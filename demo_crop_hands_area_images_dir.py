# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import torch

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

import time
import math
import json


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
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
    image_names = []
    for img_file in os.listdir(img_dir):
        if (img_file.endswith(".jpg") or 
            img_file.endswith(".jpeg") or 
            img_file.endswith(".png")):
            image_names.append(img_file)

    if len(image_names) == 0:
        print("COULD NOT FIND ANY IMAGE")

    out_dir = args.out_dir

    # Load annotation
    

    himg_count = 0
    for img_name in image_names:
        start_time = time.time()
        img = cv2.imread(os.path.join(img_dir, img_name))
        composite, predictions = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))

        composite = cv2.resize(composite, None, fx=0.5, fy=0.5)

        labels = predictions.get_field("labels").numpy().tolist()
        boxes = predictions.bbox.numpy().tolist()

        # get the keypoints
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()

        print('labels:', labels)
        print('boxes:', boxes)
        print('keypoints:', kps.shape, kps)

        # crop the hands
        if kps.shape[0] == 1:
            kps_names = PersonKeypoints.NAMES
            h1xy = kps[0, kps_names.index('left_wrist'), :2]
            h2xy = kps[0, kps_names.index('right_wrist'), :2]

            print('h1xy:', h1xy)
            print('h2xy:', h2xy)

            dist = math.sqrt((h1xy[0]-h2xy[0])**2 + (h1xy[1]-h2xy[1])**2)
            print('dist:', dist)

            hand_images = []
            # if the hands are separated, crop one image for each hand
            if dist > 400.0:
               h1 = crop_image(img, [h1xy[0], h1xy[1]], 400)
               h2 = crop_image(img, [h2xy[0], h2xy[1]], 400)
               hand_images.append(h1)
               hand_images.append(h2)
            else:
                hmiddle = [(h1xy[0] + h2xy[0])/2, (h1xy[1] + h2xy[1])/2]
                h1 = crop_image(img, hmiddle, 400)
                hand_images.append(h1)
            
            # save the hand images
            for idx, hand_image in enumerate(hand_images):
                cv2.imshow("hand " + str(idx+1) , hand_image)
                cv2.imwrite(os.path.join(out_dir, str(himg_count)+".jpg"), hand_image)
                himg_count += 1


        cv2.imshow("Detections", composite)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


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

    return img_c

if __name__ == "__main__":
    main()
