#!/usr/bin/env bash
export NGPUS=1
/home/ruichen/anaconda3/envs/maskrcnn_benchmark/bin/python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/shanghai_skus/e2e_faster_rcnn_R_50_FPN_1x_lightbox.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 4000