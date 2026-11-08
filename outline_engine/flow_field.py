"""
Edge Tangent Flow (ETF) and Coherent Directional Field Filter.

Constructs a smooth vector field of edge tangents T(x, y) = (-g_y, g_x) / ||g||
derived from image color gradients and smoothed iteratively using a bilateral
tangent kernel (Kang et al., "Coherent Line Drawing").

Filtering along this flow field aligns strokes with true object contours,
eliminating jittery high-frequency noise and producing long, graceful comic/animation lines.
"""

from typing import Tuple
import numpy as np
import cv2


class EdgeTangentFlow:
    """
    Computes an Edge Tangent Flow (ETF) unit vector field and performs
    flow-guided directional filtering along streamlines.
    """

    def __init__(self, iterations: int = 1, radius: int = 2):
        self.iterations = max(1, iterations)
        self.radius = max(1, min(4, radius))

    @staticmethod
    def _compute_multi_channel_gradients(
        img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes spatial gradients (gx, gy) and magnitude using the channel
        with the maximum gradient at each pixel (captures colored highlights and edges).
        """
        if img.ndim == 2:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            return gx, gy, mag

        # Multi-channel: compute per-channel gradients and pick the dominant channel per pixel
        h, w = img.shape[:2]
        mag_max = np.zeros((h, w), dtype=np.float32)
        gx_out = np.zeros((h, w), dtype=np.float32)
        gy_out = np.zeros((h, w), dtype=np.float32)

        for c in range(img.shape[2]):
            chan = img[:, :, c]
            cgx = cv2.Sobel(chan, cv2.CV_32F, 1, 0, ksize=3)
            cgy = cv2.Sobel(chan, cv2.CV_32F, 0, 1, ksize=3)
            cmag = np.sqrt(cgx * cgx + cgy * cgy)

            mask = cmag > mag_max
            mag_max[mask] = cmag[mask]
            gx_out[mask] = cgx[mask]
            gy_out[mask] = cgy[mask]

        return gx_out, gy_out, mag_max

    def compute_field(
        self,
        bgr_or_gray: np.ndarray,
        iterations: int | None = None,
        radius: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the smoothed 2D unit tangent vector field (tx, ty) and gradient magnitude.

        Returns:
            tx: Unit tangent X-component (-1.0 to 1.0) of shape (H, W).
            ty: Unit tangent Y-component (-1.0 to 1.0) of shape (H, W).
            mag: Gradient magnitude map normalized to 0.0 - 255.0 of shape (H, W).
        """
        num_iters = iterations if iterations is not None else self.iterations
        r = radius if radius is not None else self.radius

        h, w = bgr_or_gray.shape[:2]

        # For fast CPU throughput, compute ETF on a half-resolution grid if image is large
        downscale = max(h, w) > 320
        if downscale:
            scale = 0.5
            sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
            interp = cv2.INTER_AREA
            work_img = cv2.resize(bgr_or_gray, (sw, sh), interpolation=interp)
        else:
            work_img = bgr_or_gray

        gx, gy, mag = self._compute_multi_channel_gradients(work_img)
        eps = 1e-6

        # Initial tangent is perpendicular to gradient: (-gy, gx)
        tx = -gy / (mag + eps)
        ty = gx / (mag + eps)

        # Offsets within Euclidean disc radius r
        offsets = [
            (dy, dx)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if (dy * dy + dx * dx) <= (r * r)
        ]

        # Precompute normalized spatial distances
        spatial_weights = {
            (dy, dx): float(np.exp(-0.5 * (dy * dy + dx * dx) / (r * r)))
            for dy, dx in offsets
        }

        # Iterative Kang et al. bilateral tangent smoothing
        for _ in range(num_iters):
            sum_tx = np.zeros_like(tx)
            sum_ty = np.zeros_like(ty)

            for dy, dx in offsets:
                ws = spatial_weights[(dy, dx)]
                if dy == 0 and dx == 0:
                    q_tx, q_ty, q_mag = tx, ty, mag
                else:
                    q_tx = np.roll(np.roll(tx, dy, axis=0), dx, axis=1)
                    q_ty = np.roll(np.roll(ty, dy, axis=0), dx, axis=1)
                    q_mag = np.roll(np.roll(mag, dy, axis=0), dx, axis=1)

                dot = tx * q_tx + ty * q_ty
                phi = np.where(dot >= 0, 1.0, -1.0)
                wm = 0.5 * (1.0 + np.tanh(0.04 * (q_mag - mag)))
                wd = np.abs(dot)

                weight = ws * wm * wd
                sum_tx += phi * q_tx * weight
                sum_ty += phi * q_ty * weight

            norm = np.sqrt(sum_tx * sum_tx + sum_ty * sum_ty) + eps
            tx = sum_tx / norm
            ty = sum_ty / norm

        # Upscale back to original canvas resolution if downscaled
        if downscale:
            tx = cv2.resize(tx, (w, h), interpolation=cv2.INTER_LINEAR)
            ty = cv2.resize(ty, (w, h), interpolation=cv2.INTER_LINEAR)
            norm = np.sqrt(tx * tx + ty * ty) + eps
            tx = tx / norm
            ty = ty / norm

            # Normalize magnitude for caller use
            _, _, mag_full = self._compute_multi_channel_gradients(bgr_or_gray)
            mag = mag_full

        mag_norm = np.clip((mag / (mag.max() + eps)) * 255.0, 0, 255).astype(np.float32)
        return tx, ty, mag_norm

    def filter_along_tangents(
        self,
        img: np.ndarray,
        tx: np.ndarray,
        ty: np.ndarray,
        line_length: int = 2,
    ) -> np.ndarray:
        """
        Directionally filters an image along the streamlines of the tangent flow field (tx, ty).
        Averages noise along strokes while preserving razor-sharp edge contrast across strokes.
        """
        h, w = img.shape[:2]
        L = max(1, min(4, line_length))
        steps = list(range(-L, L + 1))

        # 1D Gaussian kernel along the streamline
        sigma = float(L) * 0.75
        raw_weights = [np.exp(-0.5 * (s / sigma) ** 2) for s in steps]
        total_w = sum(raw_weights)
        weights = [float(w_val / total_w) for w_val in raw_weights]

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)

        if img.ndim == 2:
            accum = np.zeros((h, w), dtype=np.float32)
            src_f = img.astype(np.float32)
            for step, w_val in zip(steps, weights):
                if step == 0:
                    accum += w_val * src_f
                else:
                    mx = X + float(step) * tx
                    my = Y + float(step) * ty
                    sampled = cv2.remap(
                        src_f,
                        mx,
                        my,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                    accum += w_val * sampled
            return np.clip(accum, 0, 255).astype(img.dtype)
        else:
            # Multi-channel
            accum = np.zeros_like(img, dtype=np.float32)
            src_f = img.astype(np.float32)
            for step, w_val in zip(steps, weights):
                if step == 0:
                    accum += w_val * src_f
                else:
                    mx = X + float(step) * tx
                    my = Y + float(step) * ty
                    sampled = cv2.remap(
                        src_f,
                        mx,
                        my,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                    accum += w_val * sampled
            return np.clip(accum, 0, 255).astype(img.dtype)

    @staticmethod
    def compute_step_alignment(
        tx: np.ndarray,
        ty: np.ndarray,
        curr_y: int,
        curr_x: int,
        next_y: int,
        next_x: int,
    ) -> float:
        """
        Measures the absolute cosine similarity [0.0, 1.0] between a candidate step
        from (curr_x, curr_y) to (next_x, next_y) and the coherent edge tangent flow at that location.
        """
        dx = float(next_x - curr_x)
        dy = float(next_y - curr_y)
        dist = np.hypot(dx, dy)
        if dist < 1e-4:
            return 1.0

        step_tx = dx / dist
        step_ty = dy / dist

        local_tx = float(tx[curr_y, curr_x])
        local_ty = float(ty[curr_y, curr_x])

        dot = abs(step_tx * local_tx + step_ty * local_ty)
        return float(min(1.0, dot))
