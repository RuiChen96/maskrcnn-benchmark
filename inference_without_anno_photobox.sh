#!/usr/bin/env bash

config="/extra/rui/Photobox_new30skus/e2e_faster_rcnn_R_50_FPN_1x_lightbox_new30skus.yaml"
img_dir="/extra/rui/New30skus/train/train_new30skus_500imgs/body_crop/images"
out_dir="/home/rui/photobox2real/generated_anno"

/home/rui/anaconda2/envs/maskrcnn/bin/python inference_without_anno.py  --config-file ${config} \
                                                                        --img-dir ${img_dir} \
                                                                        --out-dir ${out_dir}
