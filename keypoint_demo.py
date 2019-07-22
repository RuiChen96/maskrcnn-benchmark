from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
import cv2

config_file = "configs/e2e_keypoint_rcnn_R_50_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

# load image and then run prediction

image = cv2.imread("/home/ruichen/Documents/Documents_from_ubuntu_1604/ceiling_camera/cam29/1562086062.4555.png")

predictions = coco_demo.run_on_opencv_image(image)

print(predictions)
