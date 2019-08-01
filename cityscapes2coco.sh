#!/usr/bin/env bash
python tools/cityscapes/convert_cityscapes_to_coco.py   --dataset cityscapes_instance_only \
                                                        --outdir /home/ruichen/data/CityPersons_DATASET/annotations_coco_format \
                                                        --datadir /home/ruichen/data/CityPersons_DATASET