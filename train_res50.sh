#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python tools/train_net.py --config-file "configs/visual_pr/configs_e2e_faster_rcnn_R_50_FPN_1x_p5_ceiling.yaml"