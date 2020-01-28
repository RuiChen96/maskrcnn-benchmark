#!/usr/bin/env bash
export NGPUS=4
/home/rui/anaconda2/envs/maskrcnn/bin/python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/1000skus/multi_pr_e2e_faster_rcnn_R_101_FPN_1x_cocostyle.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000