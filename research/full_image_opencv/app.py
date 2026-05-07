import cv2
import sys
import os
import numpy as np
from pathlib import Path

# Import research methods
import grabcut
import hough_segment
import math_intersection
import contour_based
import ui_anchor

METHODS = {
    "GrabCut": grabcut.detect,
    "Hough Seg": hough_segment.detect,
    "Math Intersect": math_intersection.detect,
    "Contour": contour_based.detect,
    "UI Anchor": ui_anchor.detect
}

METHOD_NAMES = list(METHODS.keys())

state = {
    "img_paths": [],
    "idx": 0,
    "method_idx": 0,
    "img_display": None,
    "scale": 1.0
}

def load_image(idx):
    if not (0 <= idx < len(state["img_paths"])): return
    img_path = state["img_paths"][idx]
    img = cv2.imread(str(img_path))
    if img is None: return
    
    state["idx"] = idx
    h, w = img.shape[:2]
    state["scale"] = min(1600/w, 900/h, 1.0)
    state["img_display"] = cv2.resize(img, (int(w*state["scale"]), int(h*state["scale"])))
    
    # Run current method
    method_name = METHOD_NAMES[state["method_idx"]]
    corners = METHODS[method_name](img)
    
    vis = state["img_display"].copy()
    if corners:
        pts = np.array([corners["TL"], corners["TR"], corners["BR"], corners["BL"]])
        pts = (pts * state["scale"]).astype(int)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for name, pt in corners.items():
            cv2.circle(vis, (int(pt[0]*state["scale"]), int(pt[1]*state["scale"])), 5, (0, 0, 255), -1)
    
    cv2.putText(vis, f"Method: {method_name} (Press 'M' to switch)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"File: {img_path.name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Full Image Research", vis)

def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <folder_path>")
        return
    
    folder = Path(sys.argv[1])
    state["img_paths"] = sorted([f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    
    if not state["img_paths"]:
        print("No images found.")
        return

    load_image(0)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'): load_image(state["idx"] + 1)
        elif key == ord('a'): load_image(state["idx"] - 1)
        elif key == ord('m'):
            state["method_idx"] = (state["method_idx"] + 1) % len(METHOD_NAMES)
            load_image(state["idx"])
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
