#!/usr/bin/env bash
export NGPUS=4
/home/rui/anaconda2/envs/maskrcnn/bin/python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/30skus_syn2real/e2e_faster_rcnn_X_101_32x8d_FPN_1x_synth_blurry.yaml"