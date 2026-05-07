"""
Method: GrabCut Segmentation
Approach: Uses the GrabCut algorithm to separate the monitor (foreground) from the wall (background).
Pros: Good at ignoring complex UI content on the screen and focusing on the overall object mass.
"""

import cv2
import numpy as np

def detect(image):
    h_img, w_img = image.shape[:2]
    sc = 0.25
    sm = cv2.resize(image, (int(w_img * sc), int(h_img * sc)))
    sh, sw = sm.shape[:2]

    mx, my = int(sw * 0.04), int(sh * 0.07)
    rect = (mx, my, sw - 2*mx, sh - 2*my)
    mask = np.zeros((sh, sw), np.uint8)
    bg_m, fg_m = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(sm, mask, rect, bg_m, fg_m, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8), iterations=3)

    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    
    largest = max(cnts, key=cv2.contourArea)
    box = np.intp(cv2.boxPoints(cv2.minAreaRect(largest)))
    corners_orig = (box * (1.0 / sc)).astype(int)

    s = corners_orig.sum(axis=1)
    diff = np.diff(corners_orig, axis=1).flatten()
    return {
        "TL": tuple(int(x) for x in corners_orig[np.argmin(s)]),
        "TR": tuple(int(x) for x in corners_orig[np.argmin(diff)]),
        "BR": tuple(int(x) for x in corners_orig[np.argmax(s)]),
        "BL": tuple(int(x) for x in corners_orig[np.argmax(diff)])
    }
