"""
Morphological Skeletonization and Medial Axis Transform (MAT).
Converts binary ink regions into 1-pixel centerlines with exact stroke radius mapping.
"""

import cv2
import numpy as np


class StrokeSkeletonizer:
    """
    Fast vectorized Zhang-Suen morphological thinning and Medial Axis Transform.
    Uses 256-entry lookup tables and cv2.filter2D for high-speed CPU execution.
    """

    def __init__(self):
        self._lut1, self._lut2 = self._build_zhang_suen_luts()
        # 8-neighbor bitmask convolution kernel
        # P2(top)=1, P3(top-right)=2, P4(right)=4, P5(bottom-right)=8,
        # P6(bottom)=16, P7(bottom-left)=32, P8(left)=64, P9(top-left)=128
        self._kernel = np.array(
            [[128, 1, 2], [64, 0, 4], [32, 16, 8]], dtype=np.float32
        )

    @staticmethod
    def _build_zhang_suen_luts() -> tuple[np.ndarray, np.ndarray]:
        """Precomputes lookup tables for Zhang-Suen sub-iterations 1 and 2."""
        lut1 = np.zeros(256, dtype=np.uint8)
        lut2 = np.zeros(256, dtype=np.uint8)

        for v in range(256):
            # p[0]=P2, p[1]=P3, p[2]=P4, p[3]=P5, p[4]=P6, p[5]=P7, p[6]=P8, p[7]=P9
            p = [(v >> k) & 1 for k in range(8)]
            b = sum(p)  # non-zero neighbor count
            if 2 <= b <= 6:
                # Number of 0 -> 1 transitions in circular sequence
                seq = p + [p[0]]
                a = sum(1 for i in range(8) if seq[i] == 0 and seq[i + 1] == 1)
                if a == 1:
                    # Sub-iteration 1: P2*P4*P6 == 0 and P4*P6*P8 == 0
                    if (p[0] * p[2] * p[4] == 0) and (p[2] * p[4] * p[6] == 0):
                        lut1[v] = 1
                    # Sub-iteration 2: P2*P4*P8 == 0 and P2*P6*P8 == 0
                    if (p[0] * p[2] * p[6] == 0) and (p[0] * p[4] * p[6] == 0):
                        lut2[v] = 1

        return lut1, lut2

    def thin(self, binary_img: np.ndarray, max_iterations: int = 40) -> np.ndarray:
        """
        Reduces binary strokes (255 = ink, 0 = background) to 1-pixel centerlines.
        Returns a binary uint8 image (255 = skeleton, 0 = background).
        """
        curr = (binary_img > 0).astype(np.uint8)

        for _ in range(max_iterations):
            # Sub-iteration 1
            neighbors1 = cv2.filter2D(
                curr, cv2.CV_32F, self._kernel, borderType=cv2.BORDER_CONSTANT
            )
            n_idx1 = np.clip(neighbors1, 0, 255).astype(np.uint8)
            del1 = cv2.bitwise_and(cv2.LUT(n_idx1, self._lut1), curr)
            if cv2.countNonZero(del1) == 0:
                break
            curr = cv2.subtract(curr, del1)

            # Sub-iteration 2
            neighbors2 = cv2.filter2D(
                curr, cv2.CV_32F, self._kernel, borderType=cv2.BORDER_CONSTANT
            )
            n_idx2 = np.clip(neighbors2, 0, 255).astype(np.uint8)
            del2 = cv2.bitwise_and(cv2.LUT(n_idx2, self._lut2), curr)
            if cv2.countNonZero(del2) == 0:
                break
            curr = cv2.subtract(curr, del2)

        return (curr * 255).astype(np.uint8)

    def compute_medial_axis(
        self, binary_img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Medial Axis Transform (MAT).

        Returns:
            skeleton: 1-pixel binary centerline (uint8, 0 or 255).
            radius_map: Euclidean distance transform (float32) representing the
                        exact stroke half-thickness at each centerline pixel.
        """
        # Distance transform on ink pixels
        dist_map = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
        skeleton = self.thin(binary_img)
        return skeleton, dist_map
