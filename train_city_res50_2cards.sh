#!/usr/bin/env bash
export NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/cityscapes/det_e2e_faster_rcnn_R_50_FPN_1x_cocostyle_2cards.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000