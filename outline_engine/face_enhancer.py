from pathlib import Path
import urllib.request
import cv2
import numpy as np

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEFAULT_YUNET_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


class FaceEnhancer:
    """
    Lightweight facial landmark detector and anatomical feature extractor.
    Uses OpenCV's official YuNet ONNX model (~227 KB) to detect faces and 5 key landmarks:
    [right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner].
    """

    def __init__(self, model_path: Path | str = DEFAULT_YUNET_PATH):
        self.model_path = Path(model_path)
        self.detector = None
        self._current_size: tuple[int, int] | None = None

    def ensure_model(self) -> Path:
        """Downloads the 227 KB YuNet ONNX model if not already cached locally."""
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[FaceEnhancer] Downloading ultra-lightweight YuNet model (~227 KB) to {self.model_path}...")
            urllib.request.urlretrieve(YUNET_URL, str(self.model_path))
            print("[FaceEnhancer] Download complete.")
        return self.model_path

    def _get_detector(self, width: int, height: int):
        """Initializes or updates input resolution for cv2.FaceDetectorYN."""
        if self.detector is None:
            model_file = self.ensure_model()
            self.detector = cv2.FaceDetectorYN_create(
                str(model_file),
                "",
                (width, height),
                score_threshold=0.45,
                nms_threshold=0.30,
                top_k=5000,
            )
            self._current_size = (width, height)
        elif self._current_size != (width, height):
            self.detector.setInputSize((width, height))
            self._current_size = (width, height)
        return self.detector

    def detect_faces(self, bgr_img: np.ndarray) -> np.ndarray | None:
        """
        Detects faces in a BGR image.
        Returns an array of shape (N, 15) or None if no faces are found.
        Each row contains:
        [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        try:
            h, w = bgr_img.shape[:2]
            detector = self._get_detector(w, h)
            _, faces = detector.detect(bgr_img)
            return faces
        except Exception as exc:
            print(f"[FaceEnhancer] Warning: Face detection failed ({exc}). Continuing without face enhancement.")
            return None

    def extract_face_features(
        self, bgr_img: np.ndarray, detail: str = "medium"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts natural, animation-style facial lines and generates a skin suppression mask.

        Returns:
            face_lines: uint8 binary image (255 = ink stroke, 0 = background) of eyebrows,
                        eyelids, pupils, nostrils, lip contours, and jawline.
            skin_mask:  uint8 binary mask (255 = smooth skin area) covering cheeks and
                        forehead to silence random shadow scratches and gradient noise.
        """
        h, w = bgr_img.shape[:2]
        face_lines = np.zeros((h, w), dtype=np.uint8)
        skin_mask = np.zeros((h, w), dtype=np.uint8)

        faces = self.detect_faces(bgr_img)
        if faces is None or len(faces) == 0:
            return face_lines, skin_mask

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]

        # Parameters based on detail preset
        c_brow = 6 if detail != "high" else 5
        c_eye = 4 if detail != "high" else 3
        c_mouth = 6 if detail != "high" else 5

        for face in faces:
            bbox = face[0:4].astype(int)
            landmarks = face[4:14].reshape((5, 2)).astype(int)
            fx, fy, fw, fh = bbox
            re, le, nose, rm, lm = landmarks

            eye_dist = float(np.linalg.norm(le - re))
            if eye_dist < 6.0:
                continue

            # 1. Skin suppression mask: smooth cheeks, forehead, and temples
            cx = int((re[0] + le[0] + nose[0]) / 3.0)
            cy = int((re[1] + le[1] + nose[1] + rm[1] + lm[1]) / 5.0)
            rx = int(fw * 0.44)
            ry = int(fh * 0.48)
            cv2.ellipse(skin_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

            # 2. Eyebrows & Eyes
            for eye_pt in [re, le]:
                ex, ey = int(eye_pt[0]), int(eye_pt[1])

                # Eyebrow ROI (above eye landmark)
                bw = int(eye_dist * 0.58)
                bh = int(eye_dist * 0.30)
                bx1, bx2 = max(0, ex - bw // 2), min(w, ex + bw // 2)
                by1, by2 = max(0, ey - int(eye_dist * 0.48)), min(h, ey - int(eye_dist * 0.12))
                self._fit_eyebrow_spline(face_lines, gray, bx1, by1, bx2, by2, ex, ey, eye_dist)

                # Eye ROI (centered at eye landmark)
                ew = int(eye_dist * 0.52)
                eh = int(eye_dist * 0.34)
                ex1, ex2 = max(0, ex - ew // 2), min(w, ex + ew // 2)
                ey1, ey2 = max(0, ey - eh // 2), min(h, ey + eh // 2)
                self._fit_eye_features(face_lines, gray, ex1, ey1, ex2, ey2, ex, ey, eye_dist)

            # 3. Nose (nostrils & base septum accents)
            nx, ny = int(nose[0]), int(nose[1])
            self._fit_nostril_accents(face_lines, nx, ny, eye_dist)

            # 4. Mouth & Lips (parametric stoma parting line + lower lip accent)
            self._fit_mouth_features(face_lines, gray, rm, lm, eye_dist)

            # 5. Jawline & Chin
            my2 = min(h, max(int(rm[1]), int(lm[1])) + int(eye_dist * 0.25))
            jy1 = min(int(rm[1]), int(lm[1]))
            jy2 = min(h, fy + fh + int(fh * 0.08))
            jx1 = max(0, fx - int(fw * 0.08))
            jx2 = min(w, fx + fw + int(fw * 0.08))
            j_roi = gray[jy1:jy2, jx1:jx2]
            if j_roi.size > 0:
                j_blur = cv2.bilateralFilter(j_roi, 7, 50, 50)
                j_canny = cv2.Canny(j_blur, 22, 65)
                rel_my2 = max(0, my2 - jy1)
                j_canny[:rel_my2, :] = 0
                k_j = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                j_clean = cv2.dilate(j_canny, k_j)
                face_lines[jy1:jy2, jx1:jx2] = cv2.bitwise_or(face_lines[jy1:jy2, jx1:jx2], j_clean)

        # Smooth and anti-alias the extracted face features
        face_lines = cv2.GaussianBlur(face_lines, (3, 3), 0)
        _, face_lines = cv2.threshold(face_lines, 95, 255, cv2.THRESH_BINARY)

        return face_lines, skin_mask

    @staticmethod
    def _fit_eyebrow_spline(
        canvas: np.ndarray,
        gray: np.ndarray,
        bx1: int,
        by1: int,
        bx2: int,
        by2: int,
        ex: int,
        ey: int,
        eye_dist: float,
    ) -> None:
        roi = gray[by1:by2, bx1:bx2]
        if roi.size == 0:
            return
        h_roi, w_roi = roi.shape
        th = float(np.percentile(roi, 30))
        pts = np.argwhere(roi <= min(th, 100))
        if len(pts) >= 8:
            xs = pts[:, 1].astype(np.float32)
            ys = pts[:, 0].astype(np.float32)
            try:
                poly = np.polyfit(xs, ys, 2)
                eval_xs = np.linspace(0, w_roi - 1, 15)
                eval_ys = np.clip(np.polyval(poly, eval_xs), 0, h_roi - 1)
                curve_pts = np.column_stack((eval_xs + bx1, eval_ys + by1)).astype(np.int32)
                cv2.polylines(canvas, [curve_pts.reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)
                return
            except Exception:
                pass
        # Fallback anatomical arch
        eval_xs = np.linspace(bx1, bx2, 15)
        arch_h = eye_dist * 0.08
        denom = max(1.0, (bx2 - bx1) / 2.0)
        eval_ys = (by1 + by2) / 2.0 - arch_h * (1.0 - ((eval_xs - ex) / denom) ** 2)
        curve_pts = np.column_stack((eval_xs, eval_ys)).astype(np.int32)
        cv2.polylines(canvas, [curve_pts.reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)

    @staticmethod
    def _fit_eye_features(
        canvas: np.ndarray,
        gray: np.ndarray,
        ex1: int,
        ey1: int,
        ex2: int,
        ey2: int,
        ex: int,
        ey: int,
        eye_dist: float,
    ) -> None:
        roi = gray[ey1:ey2, ex1:ex2]
        if roi.size == 0:
            return
        h_roi, w_roi = roi.shape
        sample_xs = np.linspace(2, w_roi - 3, 9)
        pts = []
        for sx in sample_xs:
            x_idx = int(round(sx))
            col = roi[: max(3, h_roi * 3 // 4), x_idx]
            if len(col) > 0:
                min_y = int(np.argmin(col))
                pts.append((float(x_idx), float(min_y)))
        if len(pts) >= 4:
            pts_arr = np.array(pts)
            try:
                poly = np.polyfit(pts_arr[:, 0], pts_arr[:, 1], 2)
                eval_xs = np.linspace(2, w_roi - 3, 15)
                eval_ys = np.clip(np.polyval(poly, eval_xs), 0, h_roi - 1)
                curve_pts = np.column_stack((eval_xs + ex1, eval_ys + ey1)).astype(np.int32)
                cv2.polylines(canvas, [curve_pts.reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)
            except Exception:
                pass
        # Pupil: darkest patch
        min_val, _, min_loc, _ = cv2.minMaxLoc(roi)
        pr = int(max(2, round(eye_dist * 0.055)))
        cv2.circle(canvas, (ex1 + min_loc[0], ey1 + min_loc[1]), pr, 255, -1)

    @staticmethod
    def _fit_mouth_features(
        canvas: np.ndarray,
        gray: np.ndarray,
        rm: np.ndarray,
        lm: np.ndarray,
        eye_dist: float,
    ) -> None:
        h, w = gray.shape
        mx1 = max(0, min(int(rm[0]), int(lm[0])) - int(eye_dist * 0.12))
        mx2 = min(w, max(int(rm[0]), int(lm[0])) + int(eye_dist * 0.12))
        my1 = max(0, min(int(rm[1]), int(lm[1])) - int(eye_dist * 0.15))
        my2 = min(h, max(int(rm[1]), int(lm[1])) + int(eye_dist * 0.25))
        roi = gray[my1:my2, mx1:mx2]
        if roi.size == 0:
            return
        h_roi, w_roi = roi.shape
        sample_xs = np.linspace(int(eye_dist * 0.12), max(0, w_roi - int(eye_dist * 0.12)), 11)
        pts = []
        for sx in sample_xs:
            x_idx = int(round(sx))
            if 0 <= x_idx < w_roi:
                col = roi[:, x_idx]
                min_y = int(np.argmin(col))
                pts.append((float(x_idx), float(min_y)))
        if len(pts) >= 5:
            pts_arr = np.array(pts)
            try:
                poly = np.polyfit(pts_arr[:, 0], pts_arr[:, 1], 2)
                eval_xs = np.linspace(pts_arr[0, 0], pts_arr[-1, 0], 21)
                eval_ys = np.clip(np.polyval(poly, eval_xs), 0, h_roi - 1)
                curve_pts = np.column_stack((eval_xs + mx1, eval_ys + my1)).astype(np.int32)
                cv2.polylines(canvas, [curve_pts.reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)
            except Exception:
                pass

        # Lower lip underline accent
        cx = int((rm[0] + lm[0]) / 2.0)
        cy = int((rm[1] + lm[1]) / 2.0 + eye_dist * 0.18)
        if 0 <= cy < h and 0 <= cx < w:
            lw = max(3, int(eye_dist * 0.14))
            lip_xs = np.linspace(cx - lw, cx + lw, 11)
            lip_ys = cy + (eye_dist * 0.03) * (1.0 - ((lip_xs - cx) / float(lw)) ** 2)
            lip_pts = np.column_stack((lip_xs, lip_ys)).astype(np.int32)
            cv2.polylines(canvas, [lip_pts.reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)

    @staticmethod
    def _fit_nostril_accents(
        canvas: np.ndarray,
        nx: int,
        ny: int,
        eye_dist: float,
    ) -> None:
        h, w = canvas.shape
        nw = eye_dist * 0.12
        for side in [-1, 1]:
            nos_x = nx + side * nw
            nos_y = ny + eye_dist * 0.02
            t_nos = np.linspace(-1.0, 1.0, 7)
            xs = nos_x + t_nos * (eye_dist * 0.045)
            ys = nos_y - (eye_dist * 0.025) * (1.0 - t_nos ** 2)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if np.all(valid):
                pts = np.column_stack((xs, ys)).astype(np.int32)
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)
