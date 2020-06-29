#!/usr/bin/env bash

config="configs/shanghai_skus/e2e_faster_rcnn_R_50_FPN_1x_lightbox.yaml"
img_dir="/home/ruichen/Documents/Documents_from_ubuntu_1604/shanghai_store/unannotated_data/cropped"
out_dir="/home/ruichen/Documents/Documents_from_ubuntu_1604/shanghai_store/generated_anno"

/home/ruichen/anaconda3/envs/maskrcnn_benchmark/bin/python inference_with_label_by_name_of_folder.py    --config-file ${config} \
                                                                                                        --img-dir ${img_dir} \
                                                                                                        --out-dir ${out_dir}
