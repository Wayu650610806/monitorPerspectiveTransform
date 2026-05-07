import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# Add src to path if needed, but relative imports are better
# from src.detector import MonitorDetector
# from src.processor import rotate_image, warp_perspective, fit_to_screen
# import config

# However, for a simple delivery, keeping app.py in the root and src as a package is good.
from src.detector import MonitorDetector
from src.processor import rotate_image, warp_perspective, fit_to_screen
import config

MODES = ["manual", "paddle", "yolo"]

state = {
    "points":      [],
    "img_raw":     None,        # Original from disk
    "img_orig":    None,        # After rotation
    "img_display": None,
    "scale":       1.0,
    "img_paths":   [],
    "idx":         0,
    "mode":        "manual",
    "rotation":    0,
    "auto_bbox":   None,
}

detector = MonitorDetector()

def load_image(idx):
    if not (0 <= idx < len(state["img_paths"])):
        return
    img_path = state["img_paths"][idx]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Cannot read {img_path}")
        return
    state["img_raw"]  = img
    state["rotation"] = 0
    state["idx"]      = idx
    apply_rotation()
    print(f"\n[{idx+1}/{len(state['img_paths'])}] {img_path.name}")
    if state["mode"] in ("paddle", "yolo"):
        run_detection()

def apply_rotation():
    state["img_orig"] = rotate_image(state["img_raw"], state["rotation"])
    state["img_display"], state["scale"] = fit_to_screen(state["img_orig"])
    state["points"] = []
    state["auto_bbox"] = None
    try: cv2.destroyWindow("Warped Result")
    except: pass

def run_detection():
    mode = state["mode"]
    img = state["img_orig"]
    state["points"] = []
    state["auto_bbox"] = None

    # Use the high-level API
    warped, corners = detector.get_warped_monitor(img, mode=mode)

    if corners:
        state["points"] = [[int(k[0]), int(k[1])] for k in corners]
        # For auto_bbox, we still need to call the specific detect methods if we want the box drawn
        # or we can just modify get_warped_monitor to return the bbox too. 
        # For simplicity in the GUI, let's just get the bbox again or update the API.
        if mode == "auto": _, state["auto_bbox"] = detector.detect_paddle(img)
        else: _, state["auto_bbox"] = detector.detect_yolo(img)

    if warped is not None:
        preview, _ = fit_to_screen(warped, max_w=1280, max_h=720)
        cv2.imshow("Warped Result", preview)
    else:
        print(f"  ⚠ {mode.upper()}: No monitor detected")

def show_warped():
    # This function is now mostly handled by run_detection, 
    # but we keep it for manual mode or updates
    if len(state["points"]) == 4:
        warped = warp_perspective(state["img_orig"], state["points"])
        preview, _ = fit_to_screen(warped, max_w=1280, max_h=720)
        cv2.imshow("Warped Result", preview)


def draw_overlay():
    vis = state["img_display"].copy()
    s = state["scale"]
    
    def to_disp(p): return (int(p[0] * s), int(p[1] * s))

    # Draw BBox
    if state["auto_bbox"]:
        x1, y1, x2, y2, conf = state["auto_bbox"]
        cv2.rectangle(vis, to_disp((x1, y1)), to_disp((x2, y2)), config.BBOX_COLOR, 2)

    # Draw Points & Lines
    pts = state["points"]
    for i, pt in enumerate(pts):
        dp = to_disp(pt)
        cv2.circle(vis, dp, config.POINT_RADIUS, config.POINT_COLOR[i], -1)
        cv2.putText(vis, config.LABEL_TEXT[i], (dp[0]+10, dp[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.POINT_COLOR[i], 1)
    
    if len(pts) == 4:
        for i in range(4):
            cv2.line(vis, to_disp(pts[i]), to_disp(pts[(i+1)%4]), config.LINE_COLOR, 2)

    # UI Text
    h, w = vis.shape[:2]
    cv2.putText(vis, f"Mode: {state['mode'].upper()} | File: {state['idx']+1}/{len(state['img_paths'])}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Monitor SDK - GUI", vis)

def on_click(event, x, y, flags, param):
    if state["mode"] == "manual" and event == cv2.EVENT_LBUTTONDOWN:
        if len(state["points"]) < 4:
            ox, oy = int(x / state["scale"]), int(y / state["scale"])
            state["points"].append([ox, oy])
            if len(state["points"]) == 4: show_warped()
            draw_overlay()

def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <folder_path>")
        return
    
    folder = Path(sys.argv[1])
    state["img_paths"] = sorted([f for f in folder.iterdir() if f.suffix.lower() in config.SUPPORTED_EXT])
    
    if not state["img_paths"]:
        print("No images found.")
        return

    cv2.namedWindow("Monitor SDK - GUI")
    cv2.setMouseCallback("Monitor SDK - GUI", on_click)
    
    load_image(0)
    
    while True:
        draw_overlay()
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'): load_image(state["idx"] + 1)
        elif key == ord('a'): load_image(state["idx"] - 1)
        elif key == ord('m'):
            state["mode"] = MODES[(MODES.index(state["mode"]) + 1) % 3]
            apply_rotation()
            if state["mode"] in ("paddle", "yolo"): run_detection()
        elif key == ord('r'):
            state["rotation"] = (state["rotation"] + config.ROTATE_R_STEP) % 360
            apply_rotation()
            if state["mode"] in ("paddle", "yolo"): run_detection()
        elif key == ord('c'):
            state["rotation"] = 0
            apply_rotation()
        elif key == 32 and state["mode"] in ("paddle", "yolo"): # SPACE
            run_detection()
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
