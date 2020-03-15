#!/usr/bin/env bash

config="/home/rui/photobox2real/model_before_adapt/e2e_faster_rcnn_X_101_32x8d_FPN_1x_lightbox.yaml"
img_dir="/home/rui/real_data_skus_1-30/images"
out_dir="/home/rui/photobox2real/generated_anno"

/home/rui/anaconda2/envs/maskrcnn/bin/python inference_without_anno.py  --config-file ${config} \
                                                                        --img-dir ${img_dir} \
                                                                        --out-dir ${out_dir}
