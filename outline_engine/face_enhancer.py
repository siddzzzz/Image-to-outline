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
                brow_roi = gray[by1:by2, bx1:bx2]
                if brow_roi.size > 0:
                    b_blur = cv2.GaussianBlur(brow_roi, (5, 5), 0)
                    b_th = cv2.adaptiveThreshold(
                        b_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, c_brow
                    )
                    k_w = max(2, int(eye_dist * 0.08))
                    k_b = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 2))
                    b_clean = cv2.morphologyEx(b_th, cv2.MORPH_OPEN, k_b)
                    face_lines[by1:by2, bx1:bx2] = cv2.bitwise_or(face_lines[by1:by2, bx1:bx2], b_clean)

                # Eye ROI (centered at eye landmark)
                ew = int(eye_dist * 0.52)
                eh = int(eye_dist * 0.34)
                ex1, ex2 = max(0, ex - ew // 2), min(w, ex + ew // 2)
                ey1, ey2 = max(0, ey - eh // 2), min(h, ey + eh // 2)
                eye_roi = gray[ey1:ey2, ex1:ex2]
                if eye_roi.size > 0:
                    e_blur = cv2.GaussianBlur(eye_roi, (3, 3), 0)
                    e_th = cv2.adaptiveThreshold(
                        e_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, c_eye
                    )
                    # Filter speckles inside eye ROI, preserving coherent eyelid curves & pupil
                    num_e, labels_e, stats_e, _ = cv2.connectedComponentsWithStats(e_th, connectivity=8)
                    e_clean = np.zeros_like(e_th)
                    min_eye_comp = max(2, int(eye_dist * 0.05))
                    for i in range(1, num_e):
                        area = stats_e[i, cv2.CC_STAT_AREA]
                        w_comp = stats_e[i, cv2.CC_STAT_WIDTH]
                        if area >= min_eye_comp or w_comp >= 3:
                            e_clean[labels_e == i] = 255
                    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    e_clean = cv2.dilate(e_clean, k_e, iterations=1)
                    face_lines[ey1:ey2, ex1:ex2] = cv2.bitwise_or(face_lines[ey1:ey2, ex1:ex2], e_clean)

            # 3. Nose (nostrils & base septum curve, suppressing vertical bridge shadows)
            nx, ny = int(nose[0]), int(nose[1])
            nw, nh = int(eye_dist * 0.44), int(eye_dist * 0.28)
            nx1, nx2 = max(0, nx - nw // 2), min(w, nx + nw // 2)
            ny1, ny2 = max(0, ny - nh // 4), min(h, ny + int(nh * 0.75))
            n_roi = gray[ny1:ny2, nx1:nx2]
            if n_roi.size > 0:
                n_blur = cv2.GaussianBlur(n_roi, (3, 3), 0)
                n_th = cv2.adaptiveThreshold(
                    n_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8
                )
                face_lines[ny1:ny2, nx1:nx2] = cv2.bitwise_or(face_lines[ny1:ny2, nx1:nx2], n_th)

            # 4. Mouth & Lips (combining gray mouth parting line + Cr chromaticity contour)
            mx1 = max(0, min(int(rm[0]), int(lm[0])) - int(eye_dist * 0.16))
            mx2 = min(w, max(int(rm[0]), int(lm[0])) + int(eye_dist * 0.16))
            my1 = max(0, min(int(rm[1]), int(lm[1])) - int(eye_dist * 0.24))
            my2 = min(h, max(int(rm[1]), int(lm[1])) + int(eye_dist * 0.36))
            m_gray = gray[my1:my2, mx1:mx2]
            m_cr = cr[my1:my2, mx1:mx2]
            if m_gray.size > 0:
                m_blur = cv2.GaussianBlur(m_gray, (3, 3), 0)
                m_th1 = cv2.adaptiveThreshold(
                    m_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, c_mouth
                )
                cr_blur = cv2.GaussianBlur(m_cr, (5, 5), 0)
                cr_canny = cv2.Canny(cr_blur, 15, 45)
                m_comb = cv2.bitwise_or(m_th1, cr_canny)
                k_mw = max(2, int(eye_dist * 0.06))
                k_m = cv2.getStructuringElement(cv2.MORPH_RECT, (k_mw, 2))
                m_clean = cv2.morphologyEx(m_comb, cv2.MORPH_OPEN, k_m)
                face_lines[my1:my2, mx1:mx2] = cv2.bitwise_or(face_lines[my1:my2, mx1:mx2], m_clean)

            # 5. Jawline & Chin
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
