#!/usr/bin/env bash

config="/data/rui/syn2real/model_and_config/SKUs_1_30/e2e_faster_rcnn_X_101_32x8d_FPN_1x_synth_blurry.yaml"
img_dir="/home/rui/real_data_skus_1-30/images"
out_dir="/data/rui/syn2real/generated_anno"

/home/rui/anaconda2/envs/maskrcnn/bin/python inference_without_anno.py  --config-file ${config} \
                                                                        --img-dir ${img_dir} \
                                                                        --out-dir ${out_dir}
