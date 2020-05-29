#!/usr/bin/env bash
export NGPUS=1
/home/rui/anaconda2/envs/maskrcnn/bin/python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/photobox2picodev/e2e_faster_rcnn_R_50_pico_dev_r1500_plus_pb.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 4000