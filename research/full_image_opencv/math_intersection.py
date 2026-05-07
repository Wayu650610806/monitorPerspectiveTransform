"""
Method: Mathematical Line Intersection
Approach: Detects infinite horizontal and vertical lines at the screen's edge and calculates their intersection points.
Pros: High precision (pixel-level) and ignores internal screen content.
"""

import cv2
import numpy as np

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)
    if lines is None: return None

    h_lines, v_lines = [], []
    for line in lines:
        rho, theta = line[0]
        if abs(theta) < np.pi/18 or abs(theta - np.pi) < np.pi/18: v_lines.append((rho, theta))
        elif abs(theta - np.pi/2) < np.pi/18: h_lines.append((rho, theta))

    if len(h_lines) < 2 or len(v_lines) < 2: return None
    
    h_top, h_bot = h_lines[np.argmin([r for r, t in h_lines])], h_lines[np.argmax([r for r, t in h_lines])]
    v_left, v_right = v_lines[np.argmin([r for r, t in v_lines])], v_lines[np.argmax([r for r, t in v_lines])]

    def intersect(rho1, theta1, rho2, theta2):
        try: return tuple(int(x) for x in np.linalg.solve([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]], [rho1, rho2]))
        except np.linalg.LinAlgError: return (0, 0)

    tl = intersect(h_top[0], h_top[1], v_left[0], v_left[1])
    tr = intersect(h_top[0], h_top[1], v_right[0], v_right[1])
    br = intersect(h_bot[0], h_bot[1], v_right[0], v_right[1])
    bl = intersect(h_bot[0], h_bot[1], v_left[0], v_left[1])

    return {"TL": tl, "TR": tr, "BR": br, "BL": bl}
