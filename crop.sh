#!/usr/bin/env bash
#python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
#                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam30" \
#                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam_30_crop_800" \
#                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam30/annotations_train.json"

python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_9k/100" \
                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_9k/cam_100_crop_800" \
                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_9k/100/annotations_train.json"
