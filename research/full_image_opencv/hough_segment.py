"""
Method: Hough Line Segments
Approach: Uses HoughLinesP to find long vertical lines, filtering out UI elements (like graphs). 
Pros: Robust against screen reflections and internal UI noise.
"""

import cv2
import numpy as np

def detect(image):
    h_img, w_img = image.shape[:2]
    sc = 0.25
    sm = cv2.resize(image, (int(w_img * sc), int(h_img * sc)))
    sh, sw = sm.shape[:2]
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=200, maxLineGap=30)
    
    if lines is None: return None
    v_segs = [{'x': (l[0][0]+l[0][2])/2, 'y1': min(l[0][1], l[0][3]), 'y2': max(l[0][1], l[0][3]), 'len': np.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1])} 
              for l in lines if np.degrees(np.arctan2(abs(l[0][3]-l[0][1]), abs(l[0][2]-l[0][0]))) > 70]
              
    if not v_segs: return None
    v_segs.sort(key=lambda d: d['x'])
    groups = [[v_segs[0]]]
    for s in v_segs[1:]:
        if s['x'] - groups[-1][-1]['x'] < 35: groups[-1].append(s)
        else: groups.append([s])
        
    v_cl = [{'pos': sum(s['x']*s['len'] for s in g)/sum(s['len'] for s in g), 'items': g} for g in groups]
    lft_cands = [c for c in v_cl if sw * 0.05 < c['pos'] < sw * 0.18]
    rgt_cands = [c for c in v_cl if sw * 0.86 < c['pos'] < sw * 0.97]
    
    left_vc = max(lft_cands, key=lambda c: c['pos']) if lft_cands else None
    right_vc = min(rgt_cands, key=lambda c: c['pos']) if rgt_cands else None
    left_x = left_vc['pos'] if left_vc else sw * 0.09
    right_x = right_vc['pos'] if right_vc else sw * 0.91
    
    ref_vc = right_vc or left_vc
    top_y = min(s['y1'] for s in ref_vc['items']) if ref_vc else sh * 0.14
    bot_y = max(s['y2'] for s in ref_vc['items']) if ref_vc else sh * 0.78
    inv = 1.0 / sc
    
    return {"TL": (int(left_x * inv), int(top_y * inv)), "TR": (int(right_x * inv), int(top_y * inv)),
            "BR": (int(right_x * inv), int(bot_y * inv)), "BL": (int(left_x * inv), int(bot_y * inv))}
