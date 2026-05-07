import cv2
import numpy as np
from config import OUTPUT_W, OUTPUT_H

def rotate_image(img, angle):
    """Rotates image around center, expanding canvas to prevent cropping."""
    if angle % 360 == 0:
        return img.copy()
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0))

def warp_perspective(img, pts):
    """Applies perspective transform to the 4 given points."""
    src = np.float32(pts)
    dst = np.float32([
        [0,        0       ],
        [OUTPUT_W, 0       ],
        [OUTPUT_W, OUTPUT_H],
        [0,        OUTPUT_H],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (OUTPUT_W, OUTPUT_H))

def fit_to_screen(img, max_w=1600, max_h=900):
    """Resizes image for display purposes."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return resized, scale
