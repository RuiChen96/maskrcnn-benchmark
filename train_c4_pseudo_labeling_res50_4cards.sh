#!/usr/bin/env bash
export NGPUS=4
/home/rui/anaconda2/envs/maskrcnn/bin/python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/c4_skus/e2e_faster_rcnn_R_50_FPN_1x_carrefour1.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 4000