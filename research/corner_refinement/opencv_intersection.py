"""
Method: Smart Line Intersection
Approach: Finds horizontal and vertical lines in the corner crop and calculates their intersection.
Pros: Mathematically precise and robust to noise within the monitor screen.
"""

import cv2
import numpy as np
import math

def refine(crop, corner_type='TL'):
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=20)
    
    if lines is None: return (w//2, h//2)
    
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 30 or angle > 150: h_lines.append(line[0])
        elif 60 < angle < 120: v_lines.append(line[0])
        
    if not h_lines or not v_lines: return (w//2, h//2)
    
    # Simple intersection of first pair for research demo
    def intersect(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denom == 0: return None
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return int(px), int(py)

    for hl in h_lines:
        for vl in v_lines:
            pt = intersect(hl, vl)
            if pt and 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                return pt
                
    return (w//2, h//2)
