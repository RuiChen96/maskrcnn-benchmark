import os
import argparse
import numpy as np

import torch
import torch.utils.data as data

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.utils.data.dataloader import DataLoader


class sDataLoader(DataLoader):

    def get_stream(self):
        while True:
            for data in iter(self):
                yield data


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    # if _count_visible_keypoints(anno) >= min_keypoints_per_image:
    #     return True
    return False


class visual_pr(data.Dataset):

    classes = ('')

    def __init__(self, data_dir, split, is_training, dataset_name):
        self._split = split
        self._data_dir = data_dir
        self._data_size = -1
        self._is_training = is_training
        self._dataset_name = dataset_name

        self.ANCHORS = []

        self.classes = ('BACKGROUND',  # always index 0
                        '1_ritz_medium',
                        '2_angies_boom_pop_chip',
                        '3_red_bull_red',
                        '4_ivory_concentrated_dishwashing',
                        '5_terra_chips',
                        '6_lays_potato_chips',
                        '7_dawn_ultra_dishwashing',
                        '8_equate_cotton_bandage',
                        '9_equate_exam_gloves',
                        '10_frosted_flakes',
                        '11_red_bull_sugar_free',
                        '12_nutter_butter_cookies',
                        '13_lysol_disinfecting',
                        '14_salted_cashew_halves',
                        '15_dawn_simply_clean',
                        '16_dawn_ultra_platinum',
                        '17_oreo_cookies',
                        '18_ritz_small',
                        '19_chips_ahoy',
                        '20_vita_coconut_water',
                        '21_red_bull_blue',
                        '22_bounty_napkins',
                        '23_ritz_large',
                        '24_red_bull_yellow',
                        '25_tostitos_scoops',
                        '26_veggie_straws',
                        '27_lays_stax_chips',
                        '28_tostitos_salsa',
                        '29_tide_detergent',
                        '30_equate_wound_dressing',
                        '31_kit_kat',
                        '32_simply_orange_pulp_free',
                        '33_ruffles_original',
                        '34_maruchan_chicken',
                        '35_cheetos_crunchy',
                        '36_frosted_cheerios',
                        '37_pepperidge_farm_Goldfish',
                        '38_aquafina_water',
                        '39_doritos_cool_ranch',
                        '40_cinnamon_toast_crunch',
                        '41_Purina_alpo',
                        '42_bush_garbanzos',
                        '43_Lays_ranch_dip',
                        '44_Honey_nut_cheerios',
                        '45_Red_bull_green',
                        '46_Corn_pops',
                        '47_Apple_cheerios',
                        '48_Pure_citros_orange',
                        '49_Cocoa_puffs',
                        '50_DingDongs',
                        '51_KitKat_minis',
                        '52_Corn_pops_family',
                        '53_Jif_peanut_butter',
                        '54_Arm_hammer_carpets',
                        '55_Juicy_Juice_Apple_Buzz',
                        '56_Lucky_Charms',
                        '57_Cinnamon_Toast_Crunch',
                        '58_Pringles_Onion',
                        '59_Big_Sour_Patch',
                        '60_Donettes_Powdered',
                        '61_Starbucks_Pike_Place',
                        '62_Honey_Nut_Cheerios',
                        '63_Equate_Paper_Tape',
                        '64_Hostess_Zingers',
                        '65_Doritos_Nacho',
                        '66_Nexcare_Flexible_Tape',
                        '67_Golden_Grahams',
                        '68_Cheerios',
                        '69_Fritos_Corn_Chips',
                        '70_Reynolds_Wrap_Foil',
                        '71_Temptations',
                        '72_Trix',
                        '73_Lysol_Clean_Fresh',
                        '74_Purina_Alpo_chop',
                        '75_Diamond_Matches',
                        '76_Cheerios_small',
                        '77_Swanson_chicken',
                        '78_Arizona_green_tea',
                        '79_MM_peanut_butter',
                        '80_Multi_grain_cheerios',
                        '81_Fritos_bean_dip',
                        '82_Bandage_scissors',
                        '83_Sandies',
                        '84_Big_kit_kat',
                        '85_Vanilla_frosting',
                        '86_Lysol_hydrogen_peroxide',
                        '87_Fudge_stipes',
                        '88_Apple_jacks',
                        '89_Reeses_big_cup',
                        '90_3M_nexcare',
                        '91_Pace_picante_mild',
                        '92_Lysol_clean_fresh',
                        '93_V2_Nexcare_waterproof_tape',
                        '94_V2_American_mandarin_orange',
                        '95_American_ice',
                        '96_Bai_kula_watermelon',
                        '97_tostilos_salsa_queso',
                        '98_Twinkies',
                        '99_Coke',
                        '100_Palmolive_original',
                        '101_Dwan_orange',
                        '102_Synergy_kombucha',
                        '103_Choco_frosted_flakes',
                        '104_Peanut_MMs',
                        '105_Fabuloso_lavender',
                        '106_Cupcakes_orange',
                        '107_Palmolive_OXY',
                        '108_Moonshine_sweet_tea',
                        '109_Palmolive_orange',
                        '110_Hostess_cupcakes',
                        '111_Arz_greentea_zero',
                        '112_Gain_original',
                        '113_Fiji_water',
                        '114_Diet_coke',
                        '115_Belvita_choco',
                        '116_Pureleaf_blacktea',
                        '117_Fancy_feast_duos',
                        '118_Wishbone_italian',
                        '119_Doritos_spicy_sweet_chili',
                        '120_Pepsi_12pack',
                        '121_Cheetos_crunchy_party',
                        '122_Motts_fruit_snacks',
                        '123_Arz_mucho_mango',
                        '124_Hostess_twinkies',
                        '125_Loreal_feria',
                        '126_Tide_oxi_pods',
                        '127_Downy_unstopables',
                        '128_Fanta_orange',
                        '129_Monster_white',
                        '130_Tabasco_sauce',
                        '131_Gatorade_lemon',
                        '132_Clorox_wipes',
                        '133_Monster_black',
                        '134_Pepsi_zero_12',
                        '135_Redbull_coco',
                        '136_Cholula_sauce_org',
                        '137_Bounce_outdoor',
                        '138_Zingers',
                        '139_Dr_pepper',
                        '140_Clamato_picante',
                        '141_Diet_coke_12',
                        '142_Downy_fresh',
                        '143_Pepsi_zero',
                        '144_Ferrero_rocher',
                        '145_Diet_pepsi',
                        '146_Skinny_pop',
                        '147_Coke',
                        '148_Pepsi',
                        '149_Windex_org',
                        '150_Sprite',
                        '151_Finish',
                        '152_Clear_american',
                        '153_Fanta_straw_12',
                        '154_Glade_linen',
                        '155_Fanta_oran_12',
                        '156_Diet_coke',
                        '157_Tapatio_salsa',
                        '158_VitWater_acai',
                        '159_Cheetos_party',
                        '160_Clorox_wipes_yellow',
                        '161_Gain_org',
                        '162_Glad_press_seal',
                        '163_DrPepper_12',
                        '164_Redbull_purple',
                        '165_Dawn_ultra',
                        '166_Arz_green_tea',
                        '167_Mnt_dew',
                        '168_Motts_apple',
                        '169_Caprisun_fuit',
                        '170_Whoppers',
                        '171_Lays_BBQ',
                        '172_Sprite_12',
                        '173_Gatorade_org',
                        '174_Diet_pepsi_12',
                        '175_Milano_cookies',
                        '176_Fanta_straw',
                        '177_Revlon_colorsilk',
                        '178_Smart_water',
                        '179_KoolAid_jam_tropical',
                        '180_Raid_Ant',
                        '181_Gatorade_blue',
                        '182_Lysol_dis_blue',
                        '183_Splat_purple',
                        '184_Nutty_buddy',
                        '185_Mnt_dew_12',
                        '186_Chex_mix',
                        '187_San_pellegrino',
                        '188_Eclipse_spearmint',
                        '189_Angies_boom_yellow',
                        '190_Clorox_bleach',
                        '191_Ajax_bleach',
                        '192_Great_val_spoons',
                        '193_Quake_cakes',
                        '194_Saltine_crackers',
                        '195_Newtons',
                        '196_Dasani_water',
                        '197_Suavitel_flowers',
                        '198_Trident_spearmint',
                        '199_Eclipse_winterfrost',
                        '200_Glade_clean'
                        )

        print('Loading visual_pr data from {} ... '.format(data_dir))

        self._load()

    def _load(self, ):
        camera_view = 'train'
        test_camera_view = 'cam31'

        if self._is_training:
            annFile = os.path.join(self._data_dir, '{:s}.json'.format('annotations_' + camera_view))
        else:
            # for testing
            # annFile = os.path.join(self._data_dir, '{:s}.json'.format('annotations_coco_' + test_camera_view))
            annFile = os.path.join(self._data_dir, '{:s}.json'.format('annotations_train'))

        print('Loading annotation from {} ... '.format(annFile))

        coco = COCO(annFile)

        if not self._is_training:
            self.coco_eval = COCO(annFile)
            self.tlbr2tlwh(self.coco_eval)

        imgs = []
        for img_id in coco.imgs:
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = coco.loadAnns(ann_ids)
            # filter images without detection annotations
            if has_valid_annotation(anno):
                imgs.append((img_id, coco.imgs[img_id], 0))
        print("before remove: {}, after remove: {}. ".format(len(coco.imgs), len(imgs)))

        self._data_size = len(imgs)
        self._imgs = imgs
        self._cocos = (coco, )

        _Debug_cat_id = True
        if _Debug_cat_id:
            count_cat_id = dict()
            tmp_anns = coco.anns
            for key, val in tmp_anns.items():
                cat_id = val['category_id']
                if count_cat_id.get(cat_id, None) is not None:
                    count_cat_id[cat_id] += 1
                else:
                    count_cat_id[cat_id] = 1

        total_bbox = np.sum(list(count_cat_id.values()))
        assert total_bbox == len(coco.anns), 'bbox amount not match.'

        print("Each category contains bboxes: {}. ".format(count_cat_id))
        print("Total bboxes: {}. ".format(total_bbox))
        print("Dataset of len: {}. ".format(self.__len__()))

        return

    def _build_anchors(self):

        return []

    def _get_coco_masks(self, coco, img_id, height, width, img_name):

        return []

    def __len__(self):

        return len(self._imgs)

    def __getitem__(self, i):

        return []

    def to_detection_format(self, Dets, image_ids, im_scale_list):

        return []

    @staticmethod
    def tlbr2tlwh(coco):
        # for anno in coco.dataset['annotations']:
        #     x1, y1, x2, y2 = anno['bbox']
        #     anno['bbox'] = [x1, y1, x2 - x1, y2 - y1]

        for anno in coco.anns.values():
            x1, y1, x2, y2 = anno['bbox']
            anno['bbox'] = [x1, y1, x2 - x1, y2 - y1]

    def eval(self, result_file, start=1, end=31):
        """evaluation detection results"""
        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[1]  # specify type here
        assert self._split in ['val2014', 'minival2014', 'minival2014_new']
        cocoGt = self.coco_eval
        cocoDt = cocoGt.loadRes(result_file)

        self.tlbr2tlwh(cocoDt)

        cocoEval = COCOeval(cocoGt, cocoDt, annType)

        cocoEval.params.useCats = 1
        cocoEval.evaluate()
        cocoEval.accumulate()
        print ('With ALL CatIds...')
        cocoEval.summarize()

        # cocoEval.params.useCats = 0
        # cocoEval.evaluate()
        # cocoEval.accumulate()
        # print ('Without CatIds...')
        # cocoEval.summarize()

        cocoEval.params.useCats = 1
        for i in range(start, end):

            # catId = self._real_id_to_cat_id(i)
            catId = i

            cocoEval.params.catIds = [catId]
            cocoEval.evaluate()
            cocoEval.accumulate()
            print ('With CatIds {} {}'.format(catId, cocoGt.cats[catId]['name']))
            cocoEval.summarize()


def collate_fn(data):

    return []


def collate_fn_testing(data):

    return []


def get_loader(data_dir, split, is_training, batch_size=16, shuffle=True, num_workers=4, dataset_name=None):
    dataset = visual_pr(data_dir, split, is_training, dataset_name=dataset_name)
    if is_training:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn_testing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate per SKU AP')
    parser.add_argument('-a', '--anno-dir', action='store', dest='anno_dir', help='annotation directory')
    parser.add_argument('-b', '--bbox-dir', action='store', dest='bbox_dir', help='bounding box directory')
    parser.add_argument('-s', '--start', action='store', dest='start', help='start from SKU ...')
    parser.add_argument('-e', '--end', action='store', dest='end', help='end with SKU ...')

    args = parser.parse_args()

    # anno_dir = args.anno_dir
    # bbox_dir = args.bbox_dir
    # start = int(args.start)
    # end = int(args.end)

    anno_dir = '/Users/ruichen/bbox_trans/test_real_sku_1_30'
    bbox_dir = '/Users/ruichen/bbox_trans/syn2real_baseline/bbox_fix_done.json'
    start = 1
    end = 30

    test_set = visual_pr(anno_dir, 'minival2014_new', False, 'visual_pr')
    test_set.eval(bbox_dir, start, end)
