import urllib.request
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

U2NETP_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "u2netp.onnx"


class ForegroundSegmenter:
    """
    Lightweight foreground isolator using U2Netp (4.5 MB ONNX model) enhanced with:
    1. Multi-region horizontal tiling so people on the left/right of group photos are never missed.
    2. Fast Edge-Preserving Guided Filter so the mask snaps tightly onto real hair/face/clothing
       boundaries (eliminating any floating halo gap around subjects).
    3. Internal hole filling and uniform-background hybrid fallback for text/banners.
    """

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.session: ort.InferenceSession | None = None

    def ensure_model(self) -> Path:
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Segmenter] Downloading lightweight U2Netp model (~4.5 MB) to {self.model_path}...")
            urllib.request.urlretrieve(U2NETP_URL, str(self.model_path))
            print("[Segmenter] Download complete.")
        return self.model_path

    def _get_session(self) -> ort.InferenceSession:
        if self.session is None:
            model_file = self.ensure_model()
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 4
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(model_file),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
        return self.session

    def _run_u2netp(self, rgb_img: np.ndarray) -> np.ndarray:
        """Runs U2Netp on an RGB uint8 image and returns a float32 probability map [0.0, 1.0]."""
        h, w = rgb_img.shape[:2]
        resized = cv2.resize(rgb_img, (320, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        max_val = float(np.max(resized))
        resized /= max_val if max_val > 0 else 1.0
        tmp_img = np.zeros((320, 320, 3), dtype=np.float32)
        tmp_img[:, :, 0] = (resized[:, :, 0] - 0.485) / 0.229
        tmp_img[:, :, 1] = (resized[:, :, 1] - 0.456) / 0.224
        tmp_img[:, :, 2] = (resized[:, :, 2] - 0.406) / 0.225

        tensor = np.expand_dims(tmp_img.transpose((2, 0, 1)), axis=0).astype(np.float32)

        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})

        pred = outputs[0][:, 0, :, :]
        ma, mi = float(np.max(pred)), float(np.min(pred))
        pred = (pred - mi) / (ma - mi + 1e-8)
        prob_map = np.squeeze(pred)

        return cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_CUBIC)

    def _run_multiregion_u2netp(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Combines a full-frame pass with overlapping left/right regional passes so
        subjects standing toward the edges of group photos are captured completely.
        """
        h, w = rgb_img.shape[:2]
        p_full = self._run_u2netp(rgb_img)

        # For wider or standard aspect ratio photos, run overlapping left & right sub-windows
        if w >= int(h * 0.85):
            win_w = int(w * 0.62)
            p_left = np.zeros((h, w), dtype=np.float32)
            p_right = np.zeros((h, w), dtype=np.float32)
            p_left[:, :win_w] = self._run_u2netp(rgb_img[:, :win_w])
            p_right[:, w - win_w :] = self._run_u2netp(rgb_img[:, w - win_w :])
            return np.maximum.reduce([p_full, p_left * 0.92, p_right * 0.92])

        return p_full

    @staticmethod
    def _guided_filter(gray_guide: np.ndarray, prob: np.ndarray, radius: int = 10, eps: float = 1e-3) -> np.ndarray:
        """
        Fast O(N) Edge-Preserving Guided Filter (He et al.).
        Snaps the soft 320x320 neural probability map tightly onto the full-resolution
        edges of the original image so the outer outline hugs the subject with zero halo gap.
        """
        I = gray_guide.astype(np.float32) / 255.0
        p = prob.astype(np.float32)
        ksize = (radius, radius)

        mean_I = cv2.boxFilter(I, -1, ksize)
        mean_p = cv2.boxFilter(p, -1, ksize)
        corr_I = cv2.boxFilter(I * I, -1, ksize)
        corr_Ip = cv2.boxFilter(I * p, -1, ksize)

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, ksize)
        mean_b = cv2.boxFilter(b, -1, ksize)

        q = mean_a * I + mean_b
        return np.clip(q, 0.0, 1.0)

    def _detect_uniform_bg_mask(self, bgr_img: np.ndarray) -> tuple[bool, np.ndarray]:
        h, w = bgr_img.shape[:2]
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        border_pixels = np.concatenate([
            lab[0, :, :],
            lab[h - 1, :, :],
            lab[:, 0, :],
            lab[:, w - 1, :]
        ], axis=0)

        std_dev = float(np.mean(np.std(border_pixels, axis=0)))
        is_uniform_bg = bool(std_dev < 10.0)

        bg_median = np.median(border_pixels, axis=0)
        dist = np.linalg.norm(lab - bg_median, axis=2)

        thresh_val = max(18.0, float(np.percentile(dist, 15)) + 12.0)
        mask = (dist > thresh_val).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return is_uniform_bg, mask

    def extract_mask(
        self,
        bgr_img: np.ndarray,
        alpha_channel: np.ndarray | None = None,
        mask_threshold: float = 0.40,
    ) -> np.ndarray:
        """
        Returns a clean, edge-snapped binary uint8 mask (0 or 255) isolating all foreground entities.
        """
        h, w = bgr_img.shape[:2]

        if alpha_channel is not None and np.min(alpha_channel) < 25 and np.mean(alpha_channel > 127) > 0.02:
            mask = (alpha_channel > 110).astype(np.uint8) * 255
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return mask

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        prob_map = self._run_multiregion_u2netp(rgb_img)

        gray_guide = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        refined_prob = self._guided_filter(gray_guide, prob_map, radius=10, eps=1e-3)

        is_uniform_bg, bg_dist_mask = self._detect_uniform_bg_mask(bgr_img)
        u2_mask = (refined_prob > mask_threshold).astype(np.uint8) * 255
        u2_coverage = float(np.mean(u2_mask > 0))

        if is_uniform_bg and (u2_coverage < 0.08 or float(np.mean(bg_dist_mask > 0)) > u2_coverage * 1.15):
            raw_mask = cv2.bitwise_or(u2_mask, bg_dist_mask)
        elif u2_coverage < 0.015:
            raw_mask = bg_dist_mask
        else:
            raw_mask = u2_mask

        # Smooth mask and close narrow gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Fill small/medium internal holes inside foreground subjects (< 3% of image area)
        contours, hier = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            for i, cnt in enumerate(contours):
                if hier[0][i][3] != -1 and cv2.contourArea(cnt) < (h * w * 0.03):
                    cv2.drawContours(cleaned, [cnt], 0, 255, -1)

        cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0)
        _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)

        # Remove small disconnected background blobs (< 0.5% of image area)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        min_blob_area = max(50, int(h * w * 0.005))
        final_mask = np.zeros_like(cleaned)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_blob_area:
                final_mask[labels == idx] = 255

        if np.max(final_mask) == 0 and num_labels > 1:
            largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            final_mask[labels == largest_idx] = 255

        return final_mask
