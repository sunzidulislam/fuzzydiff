"""Create a GrabCut mask guided around the girl and dog in input.png."""
from pathlib import Path

import cv2
import numpy as np

from map import read_image, save_png


image = read_image(Path(__file__).with_name('input.png'))
guide = np.zeros(image.shape[:2], np.uint8)
# Loose silhouettes mark the subjects; GrabCut refines against photo colors.
girl = [(54, 30), (64, 17), (89, 12), (104, 22), (111, 44),
        (108, 66), (102, 84), (125, 96), (143, 117), (131, 130),
        (128, 176), (126, 224), (116, 247), (30, 247), (28, 218),
        (33, 175), (37, 128), (24, 119), (33, 100), (60, 88),
        (60, 68), (54, 49)]
dog = [(156, 153), (160, 137), (170, 146), (186, 145), (199, 148),
       (199, 161), (211, 170), (220, 187), (221, 221), (214, 241),
       (195, 243), (195, 213), (189, 210), (185, 245), (171, 245),
       (169, 219), (161, 199), (153, 181), (149, 166)]
for polygon in (girl, dog):
    cv2.fillPoly(guide, [np.array(polygon, np.int32)], 255)
kernel = np.ones((7, 7), np.uint8)
inside = cv2.erode(guide, kernel)
outside = cv2.dilate(guide, kernel)
labels = np.full(guide.shape, cv2.GC_BGD, np.uint8)
labels[outside > 0] = cv2.GC_PR_BGD
labels[guide > 0] = cv2.GC_PR_FGD
labels[inside > 0] = cv2.GC_FGD
# Keep the narrow hand regions that erosion would otherwise remove.
hands = [
    [(30, 217), (36, 218), (37, 227), (36, 242), (33, 246),
     (29, 241), (27, 234), (28, 225)],
    [(110, 220), (119, 224), (120, 232), (116, 242), (110, 244),
     (104, 240), (102, 234), (105, 227)],
]
for polygon in hands:
    cv2.fillPoly(labels, [np.array(polygon, np.int32)], cv2.GC_FGD)
hand_mask = np.zeros(image.shape[:2], np.uint8)
for polygon in hands:
    cv2.fillPoly(hand_mask, [np.array(polygon, np.int32)], 255)
out = Path(__file__).resolve().parent.parent / 'attention_results' / 'reference_style'
out.mkdir(parents=True, exist_ok=True)
save_png(out / 'hand_mask.png', hand_mask)
overlay = image.copy()
selected = hand_mask > 0
overlay[selected] = (image[selected].astype(np.float32) * 0.5 +
                     np.array([0, 255, 255], np.float32) * 0.5).astype(np.uint8)
save_png(out / 'hand_mask_overlay.png', overlay)
cv2.setRNGSeed(7)
cv2.grabCut(image, labels, None, np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64), 6, cv2.GC_INIT_WITH_MASK)
mask = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
save_png(Path(__file__).with_name('input_mask.png'), mask)
