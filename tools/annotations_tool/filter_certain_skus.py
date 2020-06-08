import os
import json
from pycocotools.coco import COCO


sku_list = [2, 8, 11, 15, 16, 17, 18, 19, 20, 23, 26, 27,
            28, 30, 32, 37, 44, 51, 92, 100, 105, 109, 116,
            149, 156, 164, 167, 176, 177, 178]

# read in annotations
anno_dir = '/Users/ruichen/Data/real_data_anno_dir'
anno_name = 'annotations_combined_1-30_200classes_v2.json'
file_path = os.path.join(anno_dir, anno_name)
with open(file=file_path, mode='r') as fp:
    anno = json.load(fp)
print('--- annotations loaded ---')

# filter out certain SKUs
filtered_anno = []
count = 0
for item in anno['annotations']:
    if item['category_id'] in sku_list:
        count += 1
    else:
        filtered_anno.append(item)

if count + len(filtered_anno) == len(anno['annotations']):
    print(count, len(filtered_anno), len(anno['annotations']))
else:
    raise ValueError

anno['annotations'] = filtered_anno
with open(file='{}/{}'.format(anno_dir, anno_name.split('.')[0] + '_filtered.json'),
          mode='w') as fp:
    json.dump(anno, fp)
print('--- re-write annotations ---')
