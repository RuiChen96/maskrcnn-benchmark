#!/usr/bin/env bash

config="configs/photobox2picodev/e2e_faster_rcnn_R_50_pico_dev_r1500_plus_pb.yaml"
img_dir="/home/rui/test_real_data_pico_dev/images_not_in_testset_crop"
out_dir="/data/rui/syn2real/generated_anno"

/home/rui/anaconda2/envs/maskrcnn/bin/python inference_without_anno.py  --config-file ${config} \
                                                                        --img-dir ${img_dir} \
                                                                        --out-dir ${out_dir}
