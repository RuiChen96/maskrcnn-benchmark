#!/usr/bin/env bash
python tools/cityscapes/convert_cityscapes_to_coco.py   --dataset cityscapes \
                                                        --outdir /extra/rui/cityscapes/cityscapes2coco \
                                                        --datadir*