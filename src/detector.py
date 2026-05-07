import cv2
import numpy as np
from pathlib import Path
from config import (
    PICODET_DIR, TINYPOSE_DIR, YOLO_MODEL_PATH, 
    USE_GPU, BBOX_EXPAND, BBOX_THRESHOLD
)

class MonitorDetector:
    def __init__(self):
        self.predictors = None
        self.yolo_model = None

    def _load_paddle(self):
        if self.predictors is not None:
            return self.predictors

        print("Loading Paddle models...")
        import paddle
        from paddle.inference import Config, create_predictor

        def _make(model_dir):
            model_dir = Path(model_dir)
            cfg = Config(str(model_dir / "model.pdmodel"),
                         str(model_dir / "model.pdiparams"))
            if USE_GPU:
                cfg.enable_use_gpu(500, 0)
            else:
                cfg.disable_gpu()
                cfg.set_cpu_math_library_num_threads(4)
            cfg.disable_glog_info()
            cfg.enable_memory_optim()
            return create_predictor(cfg)

        self.predictors = {
            "picodet":  _make(PICODET_DIR),
            "tinypose": _make(TINYPOSE_DIR),
        }
        return self.predictors

    def _load_yolo(self):
        if self.yolo_model is not None:
            return self.yolo_model
        
        from ultralytics import YOLO
        print(f"Loading YOLO model from {YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        return self.yolo_model

    def detect_paddle(self, img_bgr):
        """PicoDet + TinyPose Inference"""
        preds = self._load_paddle()
        
        # 1. PicoDet (BBox)
        p = preds["picodet"]
        h, w = img_bgr.shape[:2]
        input_size = 320
        
        img = cv2.resize(img_bgr, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)[None]

        scale_factor = np.array([[input_size / h, input_size / w]], dtype=np.float32)
        im_shape     = np.array([[h, w]], dtype=np.float32)

        input_names = p.get_input_names()
        inputs = {"image": img, "scale_factor": scale_factor, "im_shape": im_shape}
        for name in input_names:
            if name in inputs:
                t = p.get_input_handle(name)
                t.copy_from_cpu(inputs[name])

        p.run()
        out_names = p.get_output_names()
        detections = p.get_output_handle(out_names[0]).copy_to_cpu()

        if detections.shape[0] == 0:
            return None, None

        best = detections[np.argmax(detections[:, 1])]
        if best[1] < BBOX_THRESHOLD:
            return None, None

        bbox = [float(best[2]), float(best[3]), float(best[4]), float(best[5]), float(best[1])]
        
        # 2. TinyPose (Keypoints)
        p_tp = preds["tinypose"]
        input_size_tp = 512
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * BBOX_EXPAND, (y2 - y1) * BBOX_EXPAND
        side = max(bw, bh)
        bw = bh = side

        nx1, ny1 = max(0, cx - bw / 2), max(0, cy - bh / 2)
        nx2, ny2 = min(w, cx + bw / 2), min(h, cy + bh / 2)

        crop = img_bgr[int(ny1):int(ny2), int(nx1):int(nx2)]
        if crop.size == 0:
            return None, bbox
            
        crop_h, crop_w = crop.shape[:2]
        inp = cv2.resize(crop, (input_size_tp, input_size_tp))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (inp - mean) / std
        inp = inp.transpose(2, 0, 1)[None]

        p_tp.get_input_handle(p_tp.get_input_names()[0]).copy_from_cpu(inp)
        p_tp.run()
        heatmaps = p_tp.get_output_handle(p_tp.get_output_names()[0]).copy_to_cpu()

        _, K, hm_h, hm_w = heatmaps.shape
        kpts = []
        for k in range(K):
            hm = heatmaps[0, k]
            idx = np.argmax(hm)
            py, px = idx // hm_w, idx % hm_w
            kx = (px + 0.5) * input_size_tp / hm_w
            ky = (py + 0.5) * input_size_tp / hm_h
            kpts.append([float(nx1 + kx * (crop_w / input_size_tp)), 
                         float(ny1 + ky * (crop_h / input_size_tp))])

        return kpts, bbox

    def detect_yolo(self, img_bgr):
        """YOLO Pose Inference"""
        model = self._load_yolo()
        device = 0 if USE_GPU else "cpu"
        results = model(img_bgr, verbose=False, device=device)
        if not results or results[0].keypoints is None:
            return None, None

        r = results[0]
        if len(r.keypoints.xy) == 0:
            return None, None

        # Best detection by confidence
        best_idx = 0
        bbox = None
        if r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            bb = r.boxes.xyxy[best_idx].cpu().numpy()
            bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), float(confs[best_idx])]

        kpts_arr = r.keypoints.xy[best_idx].cpu().numpy()
        if kpts_arr.shape[0] < 4:
            return None, bbox

        kpts = [[float(k[0]), float(k[1])] for k in kpts_arr[:4]]
        return kpts, bbox

    def get_warped_monitor(self, img_bgr, mode='yolo'):
        """
        High-level API: Detects corners and returns the warped image.
        Returns (warped_img, corners) or (None, None) if not found.
        """
        if mode == 'yolo':
            corners, _ = self.detect_yolo(img_bgr)
        else:
            corners, _ = self.detect_paddle(img_bgr)

        if corners and len(corners) == 4:
            from .processor import warp_perspective
            warped = warp_perspective(img_bgr, corners)
            return warped, corners
        
        return None, None
