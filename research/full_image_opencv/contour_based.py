"""
Method: Area-Based / Contour Detection
Approach: Detects the monitor using area-based contour filtering, combined with HSV saturation/value masks.
Pros: Effective when the monitor is clearly separated from the background.
"""

import cv2
import numpy as np

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]

    sc  = min(0.5, 800.0 / max(h, w))
    sm  = cv2.resize(image, (int(w*sc), int(h*sc)))
    sh, sw = sm.shape[:2]
    gray_sm = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    hsv     = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    sat     = hsv[:,:,1]
    val     = hsv[:,:,2]

    blur  = cv2.bilateralFilter(gray_sm, 9, 75, 75)
    block = max(11, (int(min(sh, sw) * 0.02) // 2) * 2 + 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4)

    sv_mask = ((sat > 30) | (val < 60)).astype(np.uint8) * 255
    sv_mask[val > 215] = 0

    combined = cv2.bitwise_or(thresh, sv_mask)
    combined[:int(sh * 0.12), :] = 0

    ks = max(10, int(min(sh, sw) * 0.06))
    k  = np.ones((ks, ks), np.uint8)
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    BORDER = max(3, int(min(sh, sw) * 0.005))
    cnts_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    best_c = None
    for c in cnts_sorted:
        area = cv2.contourArea(c)
        if area < sh * sw * 0.05: break
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / max(bh, 1)
        # Check border touches
        touches = sum([y <= BORDER, y + bh >= sh - BORDER, x <= BORDER, x + bw >= sw - BORDER])
        if touches <= 1 and 0.3 < ar < 4.5:
            best_c = c
            break

    if best_c is None:
        for c in cnts_sorted:
            if cv2.contourArea(c) < sh * sw * 0.08: break
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / max(bh, 1)
            if 0.3 < ar < 4.5:
                best_c = c
                break
        if best_c is None: best_c = cnts_sorted[0]

    hull   = cv2.convexHull(best_c)
    peri   = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(-1, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(best_c)
        pts  = cv2.boxPoints(rect).astype(np.float32)

    pts  = pts / sc
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return {
        "TL": tuple(int(x) for x in pts[np.argmin(s)]),
        "TR": tuple(int(x) for x in pts[np.argmin(diff)]),
        "BR": tuple(int(x) for x in pts[np.argmax(s)]),
        "BL": tuple(int(x) for x in pts[np.argmax(diff)]),
    }
