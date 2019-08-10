#!/usr/bin/env bash
export NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/visual_pr/visual_pr_2_e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 12000