"""
Morphological Skeletonization, Medial Axis Transform, and Brush-Pen Tapering.
Extracts 1-pixel centerlines with stroke radii and brush-pen tapering.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class CenterlineStroke:
    """A continuous centerline stroke with coordinates, local radius, and loop state."""
    points: np.ndarray  # shape (N, 2) float32 coordinates (x, y)
    radii: np.ndarray   # shape (N,) float32 stroke half-width/radius
    is_closed: bool

    def get_tapered_radii(self, taper_fraction: float = 0.25, min_tip_ratio: float = 0.2) -> np.ndarray:
        """
        Applies smooth calligraphy brush-pen tapering at open stroke tips.
        Closed loops maintain their natural thickness without tip tapering.
        """
        n = len(self.points)
        if self.is_closed or n < 4:
            return self.radii.copy()

        taper_count = max(2, int(round(n * taper_fraction)))
        taper_count = min(taper_count, n // 2)

        weights = np.ones(n, dtype=np.float32)
        for i in range(taper_count):
            # Smooth Hermite/sine curve from min_tip_ratio to 1.0
            u = float(i + 1) / float(taper_count)
            t = 3.0 * u * u - 2.0 * u * u * u  # smoothstep
            factor = min_tip_ratio + (1.0 - min_tip_ratio) * t
            weights[i] = factor
            weights[-(i + 1)] = factor

        return self.radii * weights

    def extrude_tapered_ribbon(self, taper_fraction: float = 0.25) -> np.ndarray:
        """
        Extrudes the centerline along its normal vectors using tapered stroke widths.
        Returns a closed 2D polygon array of shape (2*N, 2) representing the inked brush ribbon.
        """
        n = len(self.points)
        if n < 3:
            return self.points.copy()

        pts = self.points.astype(np.float32)
        eff_radii = self.get_tapered_radii(taper_fraction=taper_fraction)

        # Compute tangent vectors along path
        tangents = np.zeros_like(pts)
        if self.is_closed:
            prev_pts = np.roll(pts, 1, axis=0)
            next_pts = np.roll(pts, -1, axis=0)
            tangents = next_pts - prev_pts
        else:
            tangents[0] = pts[1] - pts[0]
            tangents[-1] = pts[-1] - pts[-2]
            tangents[1:-1] = pts[2:] - pts[:-2]

        t_lens = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
        unit_tangents = tangents / t_lens

        # Normal vectors perpendicular to tangents: (-ty, tx)
        normals = np.column_stack([-unit_tangents[:, 1], unit_tangents[:, 0]])

        left_pts = pts + normals * eff_radii[:, None]
        right_pts = pts - normals * eff_radii[:, None]

        if self.is_closed:
            # For closed loop, return outer boundary
            return left_pts
        else:
            # For open stroke, connect left boundary forward, right boundary backward
            return np.vstack([left_pts, right_pts[::-1]])


class StrokeSkeletonizer:
    """
    Fast vectorized Zhang-Suen morphological thinning, Medial Axis Transform (MAT),
    and topological centerline stroke graph tracer.
    """

    def __init__(self):
        self._lut1, self._lut2 = self._build_zhang_suen_luts()
        # 8-neighbor bitmask convolution kernel
        self._kernel = np.array(
            [[128, 1, 2], [64, 0, 4], [32, 16, 8]], dtype=np.float32
        )
        self._nbrs = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1),
        ]

    @staticmethod
    def _build_zhang_suen_luts() -> tuple[np.ndarray, np.ndarray]:
        """Precomputes lookup tables for Zhang-Suen sub-iterations 1 and 2."""
        lut1 = np.zeros(256, dtype=np.uint8)
        lut2 = np.zeros(256, dtype=np.uint8)

        for v in range(256):
            p = [(v >> k) & 1 for k in range(8)]
            b = sum(p)
            if 2 <= b <= 6:
                seq = p + [p[0]]
                a = sum(1 for i in range(8) if seq[i] == 0 and seq[i + 1] == 1)
                if a == 1:
                    if (p[0] * p[2] * p[4] == 0) and (p[2] * p[4] * p[6] == 0):
                        lut1[v] = 1
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
            neighbors1 = cv2.filter2D(
                curr, cv2.CV_32F, self._kernel, borderType=cv2.BORDER_CONSTANT
            )
            n_idx1 = np.clip(neighbors1, 0, 255).astype(np.uint8)
            del1 = cv2.bitwise_and(cv2.LUT(n_idx1, self._lut1), curr)
            if cv2.countNonZero(del1) == 0:
                break
            curr = cv2.subtract(curr, del1)

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
            dist_map: Euclidean distance transform (float32) representing the
                      stroke radius at each centerline pixel.
        """
        dist_map = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
        skeleton = self.thin(binary_img)
        return skeleton, dist_map

    def trace_skeleton_strokes(
        self, binary_img: np.ndarray, min_length: int = 4
    ) -> list[CenterlineStroke]:
        """
        Traces a binary stroke bitmap into continuous, ordered CenterlineStroke objects
        using directional look-ahead continuity.
        """
        thin, dist = self.compute_medial_axis(binary_img)
        h, w = thin.shape
        visited = np.zeros((h, w), dtype=bool)
        b = (thin > 0).astype(np.uint8)

        # 8-neighbor degree computation
        k_deg = np.ones((3, 3), dtype=np.uint8)
        k_deg[1, 1] = 0
        deg = cv2.filter2D(b, -1, k_deg, borderType=cv2.BORDER_CONSTANT) * b

        strokes: list[CenterlineStroke] = []

        def get_unvisited_nbrs(cy: int, cx: int) -> list[tuple[int, int]]:
            res = []
            for dy, dx in self._nbrs:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and b[ny, nx] and not visited[ny, nx]:
                    res.append((ny, nx))
            return res

        def trace(sy: int, sx: int):
            if visited[sy, sx]:
                return

            pts_list = [(float(sx), float(sy))]
            rad_list = [float(dist[sy, sx])]
            visited[sy, sx] = True
            cy, cx = sy, sx
            prev_dir = None

            while True:
                candidates = get_unvisited_nbrs(cy, cx)
                if not candidates:
                    break

                if len(candidates) == 1 or prev_dir is None:
                    best_ny, best_nx = candidates[0]
                else:
                    p_dx, p_dy = prev_dir
                    best_dot = -9999.0
                    best_ny, best_nx = candidates[0]
                    for ny, nx in candidates:
                        cdx = nx - cx
                        cdy = ny - cy
                        dot = p_dx * cdx + p_dy * cdy
                        if dot > best_dot:
                            best_dot = dot
                            best_ny, best_nx = ny, nx

                prev_dir = (best_nx - cx, best_ny - cy)
                cy, cx = best_ny, best_nx
                visited[cy, cx] = True
                pts_list.append((float(cx), float(cy)))
                rad_list.append(float(dist[cy, cx]))

            if len(pts_list) >= min_length:
                dx = pts_list[0][0] - pts_list[-1][0]
                dy = pts_list[0][1] - pts_list[-1][1]
                is_closed = (len(pts_list) >= 6) and (dx * dx + dy * dy <= 4.0)
                strokes.append(
                    CenterlineStroke(
                        points=np.array(pts_list, dtype=np.float32),
                        radii=np.array(rad_list, dtype=np.float32),
                        is_closed=is_closed,
                    )
                )

        # 1. Walk endpoints first (degree 1)
        ys, xs = np.where(deg == 1)
        for sy, sx in zip(ys, xs):
            trace(int(sy), int(sx))

        # 2. Walk junctions (degree >= 3)
        ys, xs = np.where(deg >= 3)
        for sy, sx in zip(ys, xs):
            trace(int(sy), int(sx))

        # 3. Walk remaining closed loops / cycles
        rem_ys, rem_xs = np.where((b > 0) & (~visited))
        for sy, sx in zip(rem_ys, rem_xs):
            trace(int(sy), int(sx))

        return strokes
