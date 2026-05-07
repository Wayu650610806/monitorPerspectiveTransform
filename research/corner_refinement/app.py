import sys
import cv2
import numpy as np
from pathlib import Path

# Add root to path for detector and processor access
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.detector import MonitorDetector
from src.processor import rotate_image, warp_perspective, fit_to_screen
import config

# Import research methods
import opencv_features
import opencv_intersection
import cnn_refinement

MODES = ["Manual (YOLO Only)", "OpenCV Features", "Smart Intersection", "EfficientNet", "MobileNet"]
POINT_COLOR = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
LABEL_TEXT = ["TL", "TR", "BR", "BL"]

state = {
    "img_paths": [],
    "idx": 0,
    "mode_idx": 0,
    "detector": MonitorDetector(),
    "pad": 100,
    "img_raw": None,
    "img_display": None,
    "scale": 1.0,
    "yolo_points": [],
    "refined_points": [],
}

def load_image(idx):
    if not (0 <= idx < len(state["img_paths"])): return
    img_path = state["img_paths"][idx]
    img = cv2.imread(str(img_path))
    if img is None: return
    
    state["idx"] = idx
    state["img_raw"] = img
    state["img_display"], state["scale"] = fit_to_screen(img)
    
    # 1. Get initial YOLO points
    kpts, _ = state["detector"].detect_yolo(img)
    state["yolo_points"] = kpts if kpts else []
    state["refined_points"] = []
    
    run_refinement()

def run_refinement():
    mode = MODES[state["mode_idx"]]
    img = state["img_raw"]
    kpts = state["yolo_points"]
    h, w = img.shape[:2]
    
    state["refined_points"] = []
    
    if not kpts or mode == MODES[0]:
        state["refined_points"] = kpts
    else:
        try:
            temp_pts = []
            for i, (kx, ky) in enumerate(kpts):
                x1 = max(0, int(kx - state["pad"]))
                y1 = max(0, int(ky - state["pad"]))
                x2 = min(w, int(kx + state["pad"]))
                y2 = min(h, int(ky + state["pad"]))
                
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    temp_pts.append([kx, ky])
                    continue
                
                # Refine based on mode
                if mode == "OpenCV Features":
                    rx, ry = opencv_features.refine(crop)
                elif mode == "Smart Intersection":
                    rx, ry = opencv_intersection.refine(crop, LABEL_TEXT[i])
                elif mode == "EfficientNet":
                    rx, ry = cnn_refinement.refine(crop, 'EfficientNet')
                elif mode == "MobileNet":
                    rx, ry = cnn_refinement.refine(crop, 'MobileNet')
                else:
                    rx, ry = kx - x1, ky - y1 # No refinement
                
                temp_pts.append([x1 + rx, y1 + ry])
            state["refined_points"] = temp_pts
        except Exception as e:
            print(f"Error in refinement mode '{mode}': {e}")
            state["refined_points"] = kpts

    if state["refined_points"] and len(state["refined_points"]) == 4:
        warped = warp_perspective(state["img_raw"], state["refined_points"])
        preview, _ = fit_to_screen(warped, max_w=800, max_h=600)
        cv2.imshow("Warped Result (Refined)", preview)
    else:
        try: cv2.destroyWindow("Warped Result (Refined)")
        except: pass

def draw_overlay():
    vis = state["img_display"].copy()
    s = state["scale"]
    
    # Draw Refined Points (Red)
    pts = state["refined_points"]
    for i, pt in enumerate(pts):
        dp = (int(pt[0] * s), int(pt[1] * s))
        cv2.circle(vis, dp, 6, (0, 0, 255), -1)
        cv2.putText(vis, LABEL_TEXT[i], (dp[0]+10, dp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Top Bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (30, 30, 30), -1)
    mode_name = MODES[state["mode_idx"]]
    status = f"[{state['idx']+1}/{len(state['img_paths'])}] Mode: {mode_name}"
    cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Corner Refinement Research", vis)

def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <folder_path>")
        return
    
    folder = Path(sys.argv[1])
    state["img_paths"] = sorted([f for f in folder.iterdir() if f.suffix.lower() in config.SUPPORTED_EXT])
    
    if not state["img_paths"]:
        print("No images found.")
        return

    load_image(0)
    
    print("Controls:")
    print("  D/A: Next/Prev Image")
    print("  M: Switch Refinement Mode")
    print("  Q: Quit")

    while True:
        draw_overlay()
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'): load_image(state["idx"] + 1)
        elif key == ord('a'): load_image(state["idx"] - 1)
        elif key == ord('m'):
            state["mode_idx"] = (state["mode_idx"] + 1) % len(MODES)
            run_refinement()
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
