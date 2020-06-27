#!/usr/bin/env bash

config="configs/shanghai_skus/e2e_faster_rcnn_R_50_FPN_1x_lightbox.yaml"
img_dir="/extra/rui/shanghai_store_train_unannotated/cropped"
out_dir="/home/rui/photobox2real/generated_anno"

/home/rui/anaconda2/envs/maskrcnn/bin/python inference_without_anno.py  --config-file ${config} \
                                                                        --img-dir ${img_dir} \
                                                                        --out-dir ${out_dir}
