#!/usr/bin/env bash
export NGPUS=4
/home/ruichen9618/anaconda2/envs/maskrcnn_benchmark/bin/python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/1000skus/test_pr_e2e_faster_rcnn_R_101_FPN_1x_cocostyle.yaml"