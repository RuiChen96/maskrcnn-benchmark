#!/usr/bin/env bash

config="configs/c4_skus/e2e_faster_rcnn_R_50_FPN_1x_carrefour1.yaml"
img_dir="/extra/rui/c4_train_test/C4_products/train/images"
out_dir="/home/rui/photobox2real/generated_anno"

/home/rui/anaconda2/envs/maskrcnn/bin/python inference_without_anno.py  --config-file ${config} \
                                                                        --img-dir ${img_dir} \
                                                                        --out-dir ${out_dir}
