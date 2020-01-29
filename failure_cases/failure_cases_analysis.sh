#!/usr/bin/env bash

img_dir='/home/rui/real_data_skus_1-30/images'
bbox_dir='/data/rui/syn2real/generated_anno'
save_dir='/data/rui/syn2real/iter1_vis'
num_of_skus=1

/home/rui/anaconda2/envs/maskrcnn/bin/python failure_cases.py   --img-dir ${img_dir} \
                                                                --bbox-dir ${bbox_dir} \
                                                                --save-dir ${save_dir} \
                                                                --num ${num_of_skus}
