#!/usr/bin/env bash
#!/usr/bin/env bash
export NGPUS=4
/home/rui/anaconda2/envs/maskrcnn/bin/python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/domain_ad/e2e_faster_rcnn_X_101_32x8d_FPN_1x_200skus.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 4000