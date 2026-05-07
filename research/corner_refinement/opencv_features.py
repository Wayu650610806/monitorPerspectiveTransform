"""
Method: OpenCV Feature Detection
Approach: Uses Harris Corner, Shi-Tomasi, FAST, or ORB to find features in a cropped corner area.
Pros: Very fast and can find precise interest points.
"""

import cv2
import numpy as np

def refine(crop, algo='Shi-Tomasi'):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = crop.shape[:2]
    
    if algo == 'Harris':
        gray_f = np.float32(gray)
        dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        _, max_val, _, max_loc = cv2.minMaxLoc(dst)
        return max_loc
        
    elif algo == 'Shi-Tomasi':
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            return tuple(corners[0].ravel().astype(int))
            
    # Default to center if failed
    return (w//2, h//2)
