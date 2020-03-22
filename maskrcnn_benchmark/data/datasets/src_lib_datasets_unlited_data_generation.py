import glob
import os

import numpy as np
import cv2
import random
import albumentations as A

TRANSPARENT_PRODUCTS = [38, 113, 178, 196]

class unlimited_data_generator(object):
    def __init__(self, product_dir, bg_dir, hand_dir, catfile=None):
        self.hand_list = glob.glob(hand_dir + '/*')
        self.bgs = glob.glob(bg_dir + '/*')
        self.products = glob.glob(product_dir + '/*/*')
        self.max_products_one_image = 2
        
        if catfile is not None:
            categories = self.open_catfile(catfile)
            self.category_id_to_name = dict(zip(range(0, 201), categories))
        #self.minmax_size = [140, 400]
        self.minmax_size = [200, 310]
        #self.minmax_scale = [0.2, 0.7]
        #self.hand_minmax_size = [60,130]
        self.nms_threshold = 0.5
        self.BOX_COLOR = (51, 255, 51)
        self.TEXT_COLOR = (255, 255, 255)

        self.aug = self.get_aug([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
            A.augmentations.transforms.RandomGamma(),
            A.MotionBlur(blur_limit=10, p=0.4),
            A.RandomBrightnessContrast(p=0.3),
            A.augmentations.transforms.JpegCompression(quality_lower=90, p=0.3),
            A.augmentations.transforms.RandomSunFlare(num_flare_circles_lower=20, num_flare_circles_upper=30, src_radius=150, p=0.1),
            A.CLAHE(clip_limit=1.0, tile_grid_size=(10, 10), p=0.1),
            A.Posterize(num_bits=6, always_apply=False, p=0.2),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.8)
        ])

    def open_catfile(self, catfile):
        with open(catfile, 'r') as f:
            data = f.readlines()
            categories = [item.strip() for item in data]
        return categories
            
    def generate(self):
        bg_path = np.random.choice(self.bgs, 1)[0]
        bg = cv2.imread(bg_path)

        choosed_products = np.random.choice(self.products, np.random.randint(1, self.max_products_one_image))

        bbox_qualified = False
        while not bbox_qualified:
            bbox_list, category_list, image = self.generate_bbox(choosed_products, np.copy(bg))
            bbox_qualified = self.judge_generate(bbox_list)
        image = image.astype(np.uint8)

        annotations = {'image': image, 'bboxes': bbox_list, 'category_id': category_list}
        return annotations

    def augmentation(self, annotation):
        augmented = self.aug(**annotation)

        return augmented

    def merge_product(self, img, bg, scale, weight=0):
        h, w, _ = img.shape
        bg_h, bg_w, _ = bg.shape

        # black mask output size
        black_bg = np.zeros_like(bg).astype(np.uint8)

        # random location
        x_offset = np.random.randint(0, bg_w - w - 2)
        y_offset = np.random.randint(0, bg_h - h - 2)

        # copy the src image to the output black mask
        black_bg[y_offset:y_offset + h, x_offset:x_offset + w, :] = img[:, :, 0:3]

        # mask used to blackout the background
        mask = cv2.cvtColor(black_bg.astype(np.uint8), cv2.COLOR_BGR2GRAY) > 0.
        
        if weight > 0:
            # Combine product with background image
            added = cv2.addWeighted(black_bg,1.0,bg,weight,0)
            added[mask == 0] = [0, 0, 0]

            # Crop the product to src size again
            src_added = added[y_offset:y_offset + h, x_offset:x_offset + w, :]

            # new black background
            black_bg = np.zeros_like(bg).astype(np.uint8)
            # Copy the product to the black background
            black_bg[y_offset:y_offset + h, x_offset:x_offset + w, :] = src_added

            mask = cv2.cvtColor(black_bg.astype(np.uint8), cv2.COLOR_BGR2GRAY) > 0.

        bg_masked = np.copy(bg)
        bg_masked[mask != 0] = [0,0,0]
        merged = bg_masked + black_bg
    
        bbox = [x_offset, y_offset, w, h]
        
        #if np.random.uniform()>0.5:
        merged = self.merge_hand(merged, bbox, scale)       
        return merged, bbox

    def merge_hand(self, merged, bbox, scale):
        #merge hand
        x_offset, y_offset, w, h = bbox
        bg_h, bg_w, _ = merged.shape   
                     
        hand_path = np.random.choice(self.hand_list, 1)[0]
        hand = cv2.imread(hand_path)
        h_h, h_w, _ = hand.shape   
        #random_scale = np.random.randint(self.minmax_size[0], self.minmax_size[1])
        #scale = random_scale / max(h_h, h_w)
        #scale = random.uniform(self.minmax_scale[0], self.minmax_scale[1])
        hand = cv2.resize(hand, None, None, fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
        
        hand_h, hand_w, _ = hand.shape  
        hand_qualified = False
        for attemps in range(20):                           
            hand_x_offset = np.random.randint(max(x_offset-hand_w, 0)+10, min(x_offset+w+hand_w, bg_w-2)-10)
            hand_y_offset = np.random.randint(max(y_offset-hand_h, 0)+10, min(y_offset+h+hand_h, bg_h-2)-10)    
                  
            hand_bg = np.zeros_like(merged).astype(np.uint8) 
            hand_h_final = min(hand_h, bg_h-hand_y_offset)
            hand_w_final = min(hand_w, bg_w-hand_x_offset)
            hand_bg[hand_y_offset:hand_y_offset + hand_h_final, hand_x_offset:hand_x_offset + hand_w_final, :] = hand[:hand_h_final, :hand_w_final, :]
            
            bboxA = [x_offset, y_offset, x_offset+w, y_offset+h]
            bboxB = [hand_x_offset, hand_y_offset, hand_x_offset+hand_w_final, hand_y_offset+hand_h_final]
            IOU = self.bb_intersection(bboxA, bboxB)
            if IOU>0.1 and IOU<0.4:
                hand_qualified = True
                break

        if hand_qualified:               
            hand_mask = cv2.cvtColor(hand_bg.astype(np.uint8), cv2.COLOR_BGR2GRAY) > 0.     
            bg_masked = np.copy(merged)
            bg_masked[hand_mask != 0] = [0,0,0]
            merged = bg_masked + hand_bg
                        
        #hand_dil_mask = np.expand_dims(hand_mask, 2)               
        #merged = merged * (1 - hand_dil_mask) + hand_bg * hand_dil_mask
        
        return merged
        
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def bb_intersection(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea)

        # return the intersection over union value
        return iou
            
    def non_max_suppression(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        # overlapThresh is the value above which a bounding box stays and below which, the bounding box is considered redundant
        if len(boxes) == 0:
            return []
        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes, then initialize
            # the suppression list (i.e. indexes that will be deleted)
            # using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]
            # loop over all indexes in the indexes list
            for pos in range(0, last):
                # grab the current index
                j = idxs[pos]

                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]
                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap > overlapThresh:
                    suppress.append(pos)

            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
        return boxes[pick]

    def generate_bbox(self, image_list, bg):
        bbox_list = []
        category_list = []
        for image_path in image_list:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # crop the image
            #img = self.crop_bgra_img(img)

            p_h, p_w, _ = img.shape
            random_scale = np.random.randint(self.minmax_size[0], self.minmax_size[1])
            scale = random_scale / max(p_h, p_w)
            #scale = random.uniform(self.minmax_scale[0], self.minmax_scale[1])
            img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
            
            category = int(image_path.rsplit('/', 2)[-2].split('.', 1)[0])
            category_list.append(category)

            # if the product is a water bottle, add some transparency
            transparency = 0
            if category in TRANSPARENT_PRODUCTS:
                transparency = 0.4

            bg, bbox = self.merge_product(img, bg, scale, weight=transparency)
            bbox_list.append(bbox)
        return bbox_list, category_list, bg

    def crop_bgra_img(self, image):
        a = image[:, :, 3]
        rect = cv2.boundingRect(a)
        img = image[:,:,0:3]
        img = img[rect[0]:(rect[0]+rect[2]), rect[1]:(rect[1]+rect[3])]
        return img


    def judge_generate(self, bbox_list):
        len_bbox_before_nms = len(bbox_list)
        bbox_x1y1x2y2 = np.zeros_like(np.array(bbox_list))
        bbox_x1y1x2y2[:, :2] = np.array(bbox_list)[:, :2]
        bbox_x1y1x2y2[:, 2:4] = np.array(bbox_list)[:, :2] + np.array(bbox_list)[:, 2:4]
        len_after_nms = len(self.non_max_suppression(bbox_x1y1x2y2, self.nms_threshold))
        if len_after_nms != len_bbox_before_nms:
            return False
        else:
            return True

    def visualize_bbox(self, img, bbox, class_id, class_idx_to_name, thickness=2):
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=self.BOX_COLOR, thickness=thickness)
        class_name = class_idx_to_name[class_id]
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height * 2)), (x_min + text_width * 2, y_min), self.BOX_COLOR)
        cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height * 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * 2, self.TEXT_COLOR, lineType=cv2.LINE_AA)
        return img

    def visualize(self, annotations):
        img = annotations['image'].copy()
        for idx, bbox in enumerate(annotations['bboxes']):
            print('bbox:', bbox)
            img = self.visualize_bbox(img, bbox, annotations['category_id'][idx], self.category_id_to_name)
        return img

    def get_aug(self, aug, min_area=0., min_visibility=0.):
        return A.Compose(aug, A.BboxParams(format='coco', min_area=min_area,
                                           min_visibility=min_visibility,
                                           label_fields=['category_id']))


if __name__ == "__main__":
    product_dir = '/datasets3/data/ceiling/box/pilot1_images'
    bg_dir = '/datasets3/data/ceiling/background_crops/backgrounds_lab_nano_400'
    hand_dir = '/datasets3/data/ceiling/box/hand_images'
    catfile = '/datasets3/data/ceiling/categories200.txt'
    online_generator = unlimited_data_generator(product_dir, bg_dir, hand_dir, catfile)

    delay = 1
    while True:
        annotation = online_generator.generate()
        #image = annotation['image']
        #image = online_generator.visualize(annotation)
        augmented = online_generator.augmentation(annotation)
        image = online_generator.visualize(augmented)
        cv2.imshow('generated pr image', image)

        k = cv2.waitKey(delay)

        if k == 27:         # If escape was pressed exit
            cv2.destroyAllWindows()
            break
        elif k == ord('p'):
            delay = 1 - delay
