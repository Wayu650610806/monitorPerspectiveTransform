"""
Method: UI Anchor Locking
Approach: Detects specific UI elements (like colored tabs or buttons) to anchor the monitor's coordinates.
Pros: Reliable if the screen layout and brand are consistent.
"""

import cv2
import numpy as np

def detect(image):
    h_img, w_img = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Find Warm Anchor (Left/Bottom)
    warm = cv2.inRange(hsv, (0, 60, 80), (42, 255, 255))
    warm[:int(h_img * 0.30), :] = 0
    cnts, _ = cv2.findContours(cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((5,5), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 30]
    cands = [r for r in rects if r[1] + r[3] > h_img * 0.55 and r[0] < w_img * 0.60]
    wx, wy, ww, wh = min(cands, key=lambda x: x[0]) if cands else (int(w_img * 0.07), int(h_img * 0.88), int(w_img * 0.04), 0)
    left, bottom = wx, wy + wh

    # 2. Right Edge Scan
    band = gray[max(0, bottom - int(h_img * 0.08)//2):min(h_img, bottom + int(h_img * 0.08)//2), left + ww * 5:]
    right = left + ww * 5 + int(np.where(np.convolve((band.mean(axis=0) < 90).astype(float), np.ones(5)/5, 'same') > 0.6)[0].max()) if band.size else int(w_img * 0.90)

    # 3. Top Teal Tab
    teal_roi = np.zeros_like(gray)
    teal_roi[:int(h_img * 0.55), left:right] = 255
    teal = cv2.bitwise_and(cv2.inRange(hsv, (75, 60, 60), (105, 255, 255)), teal_roi)
    t_cnts, _ = cv2.findContours(cv2.morphologyEx(teal, cv2.MORPH_CLOSE, np.ones((5,20), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top = min([cv2.boundingRect(c)[1] for c in t_cnts if cv2.boundingRect(c)[2] > (right-left)*0.10], default=max(0, bottom - int((right-left) * 0.75)))

    return {"TL": (left, top), "TR": (right, top), "BR": (right, bottom), "BL": (left, bottom)}
