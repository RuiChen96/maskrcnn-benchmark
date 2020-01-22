import os
import json

from tqdm import tqdm
from pycocotools.coco import COCO


def main():
    bbox_dir = '/home/ruichen/Documents/Documents_from_ubuntu_1604/AiFi_model_save/1000skus/synthetic_1000skus_test'
    bbox = os.path.join(bbox_dir, 'bbox_modify.json')
    # coco = COCO(bbox)
    with open(bbox, 'r') as fp:
        bbox_json = json.load(fp)
    print("--- json file loaded")
    print("--- with length:", len(bbox_json))

    pbar = tqdm(total=len(bbox_json))

    for prediction in bbox_json:
        # step 1: xywh to xyxy
        x, y, w, h = prediction["bbox"]
        prediction["bbox"] = [x, y, x + w, y + h]
        # step 2: if id >= 52, id --
        if prediction["category_id"] >= 52:
            prediction["category_id"] -= 1

        pbar.update(1)

    pbar.close()

    with open('{}/bbox_fix_done.json'.format(bbox_dir), 'w') as output_json_file:
        json.dump(bbox_json, output_json_file)

    print("--- bbox fix saved")
    print("--- with length:", len(bbox_json))


if __name__ == '__main__':
    main()
