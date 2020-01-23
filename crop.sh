#!/usr/bin/env bash
#python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
#                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam31" \
#                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam_31_crop_800" \
#                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam31/annotations_train.json"

#python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
#                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_9k/100" \
#                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_9k/cam_100_crop_800" \
#                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_9k/100/annotations_train.json"

#python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
#                                            --img-dir "/data/ubuntu/root/home/ruichen/Documents/ceiling_camera_13k/30" \
#                                            --out-dir "/data/ubuntu/root/home/ruichen/Documents/ceiling_camera_13k/cam_30_crop_800_imgs_13k" \
#                                            --annotations-file "/data/ubuntu/root/home/ruichen/Documents/ceiling_camera_13k/cam_30_crop_800_imgs_13k/annotations_train.json"

#python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
#                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_cam29_v2/29" \
#                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_cam29_v2/cam_29_crop_800_imgs_13k" \
#                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_cam29_v2/29/annotations_train.json"

#python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
#                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_cam31_v2/31" \
#                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_cam31_v2/cam_31_crop_800_imgs_13k" \
#                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera_cam31_v2/31/annotations_train.json"

#python demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
#                                            --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/hand_crop_test_data/test_cam_31_batch2/cam31" \
#                                            --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/hand_crop_test_data/test_cam_31_batch2/test_cam_31_crop_800_batch2" \
#                                            --annotations-file "/home/ruichen/Documents/Documents_from_ubuntu_1604/hand_crop_test_data/test_cam_31_batch2/cam31/annotations_train.json"

python no_anno_demo_crop_hands_area_images_dir.py   --config-file "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml" \
                                                    --img-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/Uncropped_data/originals" \
                                                    --out-dir "/home/ruichen/Documents/Documents_from_ubuntu_1604/Uncropped_data/cropped"
