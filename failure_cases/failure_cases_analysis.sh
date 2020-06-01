#!/usr/bin/env bash

#img_dir='/Users/ruichen/Data/test_real_data_pico_dev/frames4'
#bbox_dir='/Users/ruichen/Data/test_real_data_pico_dev/frames4_adaptation_results'
#save_dir='/Users/ruichen/Data/test_real_data_pico_dev/frames4_adaptation_results/failure_cases_real'
#num_of_skus=1

img_dir='/Users/ruichen/Data/test_new30skus_data/images'
bbox_dir='/Users/ruichen/Data/test_new_30skus_photobox_results/fix_all_2eps'
save_dir='/Users/ruichen/Data/test_new_30skus_photobox_results/fix_all_2eps/failure_cases_real'
num_of_skus=1

python failure_cases_analysis.py  --img-dir ${img_dir} \
                                  --bbox-dir ${bbox_dir} \
                                  --save-dir ${save_dir} \
                                  --num ${num_of_skus}
