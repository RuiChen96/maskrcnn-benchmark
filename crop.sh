#!/usr/bin/env bash
python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam29" \
                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam_29_final" \
                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/annotations_coco_cam29.json"
