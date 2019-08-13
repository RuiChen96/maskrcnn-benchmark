#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file "configs/visual_pr/configs_e2e_faster_rcnn_X_101_32x8d_FPN_1x_p5_ceiling.yaml"