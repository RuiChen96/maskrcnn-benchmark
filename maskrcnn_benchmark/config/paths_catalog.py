# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        # do not contains "coco", otherwise it will use coco dataloader
        "cityscapes_fine_instanceonly_det_train": {
            "img_dir": "cityscapes/leftImg8bit/train",
            "ann_file": "cityscapes/annotations_coco_format/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_det_val": {
            "img_dir": "cityscapes/leftImg8bit/val",
            "ann_file": "cityscapes/annotations_coco_format/instancesonly_filtered_gtFine_val.json"
        },
        "foggy_fine_instanceonly_det_train": {
            "img_dir": "foggy_cityscapes/leftImg8bit_foggy/train",
            "ann_file": "foggy_cityscapes/annotations_coco_format/instancesonly_filtered_gtFine_train.json"
        },
        "foggy_fine_instanceonly_det_val": {
            "img_dir": "foggy_cityscapes/leftImg8bit_foggy/val",
            "ann_file": "foggy_cityscapes/annotations_coco_format/instancesonly_filtered_gtFine_val.json"
        },
        "visual_pr_30skus_cam100_9k_imgs": {
            "img_dir": "visual_pr_ceiling/cam100",
            "ann_file": "visual_pr_ceiling/cam100/annotations_train.json"
        },
        "visual_pr_30skus_cam29_13k_imgs": {
            "img_dir": "visual_pr_ceiling/cam29",
            "ann_file": "visual_pr_ceiling/cam29/annotations_train.json"
        },
        "visual_pr_30skus_cam30_13k_imgs": {
            "img_dir": "visual_pr_ceiling/cam30",
            "ann_file": "visual_pr_ceiling/cam30/annotations_train.json"
        },
        "visual_pr_30skus_cam31_13k_imgs": {
            "img_dir": "visual_pr_ceiling/cam31",
            "ann_file": "visual_pr_ceiling/cam31/annotations_train.json"
        },
        "visual_pr_30skus_shelf_dark_4k_imgs": {
            "img_dir": "visual_pr_shelf/dark",
            "ann_file": "visual_pr_shelf/dark/annotations_train.json"
        },
        "visual_pr_30skus_shelf_light_2k_imgs_cam1": {
            "img_dir": "visual_pr_shelf/light/cam1",
            "ann_file": "visual_pr_shelf/light/cam1/annotations_train.json"
        },
        "visual_pr_30skus_shelf_light_2k_imgs_cam2": {
            "img_dir": "visual_pr_shelf/light/cam2",
            "ann_file": "visual_pr_shelf/light/cam2/annotations_train.json"
        },
        "visual_pr_30skus_green_screen_data_v4": {
            "img_dir": "visual_pr_syn/v4",
            "ann_file": "visual_pr_syn/v4/annotations_train.json"
        },
        "test_visual_pr_30skus_cam29_batch2_imgs": {
            "img_dir": "test_visual_pr_ceiling/cam29/batch2",
            "ann_file": "test_visual_pr_ceiling/cam29/batch2/annotations_test.json"
        },
        "test_visual_pr_30skus_cam30_batch2_imgs": {
            "img_dir": "test_visual_pr_ceiling/cam30/batch2",
            "ann_file": "test_visual_pr_ceiling/cam30/batch2/annotations_test.json"
        },
        "synthetic_1000skus_train": {
            "img_dir": "synthetic1000/images",
            "ann_file": "synthetic1000/annotations.json"
        },
        "synthetic_1000skus_test": {
            "img_dir": "synthetic1000_test/images",
            "ann_file": "synthetic1000_test/annotations.json"
        },
        "unlabeled_train_data_nano": {
            "img_dir": "unlabeled_train_data_nano/cropped",
            "ann_file": "unlabeled_train_data_nano/annotations_domain_adaptaion_v2.json"
        },
        "real_skus_1_30_train": {
            "img_dir": "real_data_skus_1-30/images",
            "ann_file": "real_data_skus_1-30/annotations_1_30_200classes.json"
        },
        "test_data_nano": {
            "img_dir": "test_data_nano/100-flipped",
            "ann_file": "test_data_nano/annotations_100-flipped_42_crop_550.json"
        },
        "synthetic1000_train_multi": {
            "img_dir": "synthetic1000_train_multi/view",
            "ann_file": "synthetic1000_train_multi/COCO.json"
        },
        "synthetic1000_test_multi": {
            "img_dir": "synthetic1000_test_multi/view",
            "ann_file": "synthetic1000_test_multi/COCO.json"
        },
        "test_real_sku_1_30": {
            "img_dir": "test_real_sku_1_30/v2_Dec12/images_30_100",
            "ann_file": "test_real_sku_1_30/v2_Dec12/ann_test_1_30_cams30-100_200classes.json"
        },
        "real_skus_1_30_train_syn2real": {
            "img_dir": "real_data_skus_1-30/images",
            "ann_file": "real_data_skus_1-30/annotations_syn2real_iter1.json"
        },
        "real_skus_1_30_train_photobox2real": {
            "img_dir": "real_data_skus_1-30/images",
            "ann_file": "real_data_skus_1-30/annotations_photobox2real_iter1.json"
        },
        "test_photobox": {
            "img_dir": "pilot1_photobox/test_pilot1_photobox/images",
            "ann_file": "pilot1_photobox/test_pilot1_photobox/annotations.json"
        },
        "real_skus_1_30_train_18skus_photobox2real": {
            "img_dir": "train_real_pilot1_photobox/images",
            "ann_file": "train_real_pilot1_photobox/annotations_photobox2real_new_18skus_iter1_0.98.json"
        },
        "sku_box_online": {
            "product_dir": "photobox_train/pilot3_30skusV2",
            "bg_dir": "photobox_train/backgrounds_lab_nano_400",
            "hand_dir": "photobox_train/hand_images",
            "catfile": "photobox_train/categories200.txt"
        },
        "sku_box_online_shanghai": {
            "product_dir": "shanghai_store_photobox/shanghai_photobox_v1",
            "bg_dir": "shanghai_store_photobox/backgrounds_lab_nano_400",
            "hand_dir": "shanghai_store_photobox/hand_images",
            "catfile": "shanghai_store_photobox/categories_shanghai_30skus.txt"
        },
        "test_photobox_hand_zoom": {
            "img_dir": "Testing_data_hand_zoom/images",
            "ann_file": "Testing_data_hand_zoom/ann_pilot1-hands-area.json"
        },
        "real_skus_1_30_train_2cls": {
            "img_dir": "real_data_skus_1-30/images",
            "ann_file": "real_data_skus_1-30/annotations_1_30_200classes.json"
        },
        "real_skus_1_30_train_filtered_2cls": {
            "img_dir": "real_data_skus_1-30/images",
            "ann_file": "real_data_skus_1-30/annotations_combined_1-30_200classes_v2_filtered_2_cls.json"
        },
        "real_skus_31_100_train_2cls": {
            "img_dir": "real_data_skus_31-100/images",
            "ann_file": "real_data_skus_31-100/annotations_combined_31-100_200-classes.json"
        },
        "real_skus_31_100_train_filtered_2cls": {
            "img_dir": "real_data_skus_31-100/images",
            "ann_file": "real_data_skus_31-100/annotations_31_100_200classes_v6_filtered_2_cls.json"
        },
        "real_skus_101_200_train_filtered_2cls": {
            "img_dir": "real_data_skus_101_200/images",
            "ann_file": "real_data_skus_101_200/annotations_skus_101_200_train_v4_filtered_2_cls.json"
        },
        "test_real_data_skus_31-100": {
            "img_dir": "test_real_data_skus_31-100/images_100",
            "ann_file": "test_real_data_skus_31-100/ann_test_31_100_cam_100.json"
        },
        "test_photobox2real_new_30skus": {
            "img_dir": "New30skus/images",
            "ann_file": "New30skus/annotations_combined_200skus.json"
        },
        "train_photobox2real_new_30skus_500imgs": {
            "img_dir": "train_new30skus_500imgs/images",
            "ann_file": "train_new30skus_500imgs/annotations_photobox2real_new_30skus_iter1_0.98.json"
        },
        "test_real_data_skus_1-30_2_cls": {
            "img_dir": "test_real_data_skus_1_30/images_30_100",
            "ann_file": "test_real_data_skus_1_30/ann_test_1_30_cams30-100_200classes_2_cls.json"
        },
        "test_real_data_skus_31-100_2_cls": {
            "img_dir": "test_real_data_skus_31-100/images_100",
            "ann_file": "test_real_data_skus_31-100/ann_test_31_100_cam_100_2_cls.json"
        },
        "test_real_data_skus_101-200_2_cls": {
            "img_dir": "test_real_data_skus_101_200/images",
            "ann_file": "test_real_data_skus_101_200/ann_test_101_200_200classes_dec_2019_2_cls.json"
        },
        "test_real_data_pico_dev_2_cls_frame_1": {
            "img_dir": "test_real_data_pico_dev/frames1",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames1_crop_2_cls.json"
        },
        "test_real_data_pico_dev_frame_1": {
            "img_dir": "test_real_data_pico_dev/frames1",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames1_crop.json"
        },
        "test_real_data_pico_dev_2_cls_frame_2": {
            "img_dir": "test_real_data_pico_dev/frames2",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames2_crop_2_cls.json"
        },
        "test_real_data_pico_dev_frame_2": {
            "img_dir": "test_real_data_pico_dev/frames2",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames2_crop.json"
        },
        "test_real_data_pico_dev_2_cls_frame_3": {
            "img_dir": "test_real_data_pico_dev/frames3",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames3_crop_2_cls.json"
        },
        "test_real_data_pico_dev_frame_3": {
            "img_dir": "test_real_data_pico_dev/frames3",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames3_crop.json"
        },
        "test_real_data_pico_dev_2_cls_frame_4": {
            "img_dir": "test_real_data_pico_dev/frames4",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames4_crop_2_cls.json"
        },
        "test_real_data_pico_dev_frame_4": {
            "img_dir": "test_real_data_pico_dev/frames4",
            "ann_file": "test_real_data_pico_dev/annotations_picodev_frames4_crop.json"
        },
        "train_real_data_pico_dev_images_not_in_testset_crop": {
            "img_dir": "test_real_data_pico_dev/images_not_in_testset_crop",
            "ann_file": "test_real_data_pico_dev/annotations_photobox2picodev_r1500_30skus_iter1_0.98.json"
        },
        "shanghai_store_train_size_450": {
            "img_dir": "/extra/rui/shanghai_store_train/size_450",
            "ann_file": "/extra/rui/shanghai_store_train/annotations_amcrest_skus1-30-hcrop_crop_450_2_cls.json"
        },
        "shanghai_store_train_unannotated": {
            "img_dir": "shanghai_store/unannotated_data/cropped/all_imgs",
            "ann_file": "shanghai_store/generated_anno/annotations_shanghai_iter1_0.98.json"
        },
        "shanghai_store_test_size_512_centered": {
            "img_dir": "shanghai_store_test/centered/amcrest",
            "ann_file": "shanghai_store_test/centered/annotations_amcrest_shangahi-test-centered-hand-crop-size512_crop_512.json"
        },
        "shanghai_store_test_size_512_non_centered": {
            "img_dir": "shanghai_store_test/non-centered/amcrest",
            "ann_file": "shanghai_store_test/non-centered/annotations_amcrest_shangahi-test-non-centered-hand-crop-size512_crop_512.json"
        },
        "test_c4_skus": {
            "img_dir": "C4_products/test/images",
            "ann_file": "C4_products/test/annotations_amcrest_Nanostore-test-non-centered-hand-crop-size512_crop_512.json"
        },
        "train_c4_skus": {
            "img_dir": "C4_products/train/images",
            "ann_file": "C4_products/train/annotations_amcrest_Nanostore-train-non-centered-hand-crop-size512_crop_512.json"
        },
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="Cityscapes",
                args=args,
            )
        elif "visual_pr" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args,
            )
        elif "foggy" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="Foggy_Cityscapes",
                args=args,
            )
        elif "1000skus" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args,
            )
        elif "nano" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_1_30_train" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "synthetic1000_train_multi" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "synthetic1000_test_multi" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_sku_1_30" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_1_30_train_syn2real" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_1_30_train_photobox2real" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_1_30_train_18skus_photobox2real" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_photobox" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "sku_box_online" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                skus_dir=os.path.join(data_dir, attrs["product_dir"]),
                bgs_dir=os.path.join(data_dir, attrs["bg_dir"]),
                hands_dir=os.path.join(data_dir, attrs["hand_dir"]),
                # catfile=os.path.join(data_dir, attrs["catfile"]),
            )
            return dict(
                factory="SKUsBoxOnlineDataset",
                args=args
            )
        elif "sku_box_online_shanghai" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                skus_dir=os.path.join(data_dir, attrs["product_dir"]),
                bgs_dir=os.path.join(data_dir, attrs["bg_dir"]),
                hands_dir=os.path.join(data_dir, attrs["hand_dir"]),
                # catfile=os.path.join(data_dir, attrs["catfile"]),
            )
            return dict(
                factory="SKUsBoxOnlineDataset",
                args=args
            )
        elif "test_photobox_hand_zoom" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_1_30_train_2cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR_2cls",
                args=args
            )
        elif "real_skus_31_100_train_2cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR_2cls",
                args=args
            )
        elif "test_real_data_skus_31-100" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_skus_1-30_2_cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_skus_31-100_2_cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_skus_101-200_2_cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_photobox2real_new_30skus" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "train_photobox2real_new_30skus_500imgs" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_2_cls_frame_1" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_frame_1" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_2_cls_frame_2" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_frame_2" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_2_cls_frame_3" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_frame_3" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_2_cls_frame_4" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_real_data_pico_dev_frame_4" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "train_real_data_pico_dev_images_not_in_testset_crop" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "shanghai_store_train_size_450" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "shanghai_store_train_unannotated" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "shanghai_store_test_size_512_centered" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "shanghai_store_test_size_512_non_centered" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_1_30_train_filtered_2cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_31_100_train_filtered_2cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "real_skus_101_200_train_filtered_2cls" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "train_c4_skus" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        elif "test_c4_skus" == name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="VisualPR",
                args=args
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url


if __name__ == '__main__':
    test = DatasetCatalog()
    test.get("synthetic_1000skus_test")
