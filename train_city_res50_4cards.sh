#!/usr/bin/env bash
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/cityscapes/det_e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000