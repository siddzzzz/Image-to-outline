"""
Adaptive Curvature Analysis, Sharp Corner Detection, and Schneider Bézier Curve Fitting.

Provides:
1. Discrete turning angle and curvature analysis to identify sharp corners (lapels, collars, serifs).
2. Philip J. Schneider's least-squares cubic Bézier curve fitting with adaptive subdivision.
3. Collinear straight-line simplification (substituting verbose 'C' curves with compact 'L' lines).
"""

from typing import Tuple, List, Optional
import numpy as np


class CurvatureAnalyzer:
    """
    Analyzes discrete curvature along 2D point sequences and detects sharp corners (C^0 cusps).
    """

    @staticmethod
    def detect_corners(
        points: np.ndarray,
        is_closed: bool = False,
        corner_thresh_deg: float = 50.0,
        k: int = 2,
    ) -> List[int]:
        """
        Detects indices of sharp corner vertices using chord-based turning angle analysis
        and 1D non-maximum suppression.

        Returns:
            List of integer point indices where a sharp corner occurs.
        """
        n = len(points)
        if n < 4:
            return []

        # Chord span k
        k = max(1, min(k, (n - 1) // 3))
        angles = np.zeros(n, dtype=np.float32)

        start_idx = 0 if is_closed else k
        end_idx = n if is_closed else (n - k)

        for i in range(start_idx, end_idx):
            p_prev = points[(i - k) % n] if is_closed else points[i - k]
            p_curr = points[i]
            p_next = points[(i + k) % n] if is_closed else points[i + k]

            v_in = p_curr - p_prev
            v_out = p_next - p_curr

            l_in = np.hypot(v_in[0], v_in[1])
            l_out = np.hypot(v_out[0], v_out[1])

            if l_in > 1e-4 and l_out > 1e-4:
                u_in = v_in / l_in
                u_out = v_out / l_out
                dot = np.clip(np.dot(u_in, u_out), -1.0, 1.0)
                angles[i] = float(np.degrees(np.arccos(dot)))

        # Non-maximum suppression over window of +/- k
        corners = []
        for i in range(start_idx, end_idx):
            if angles[i] >= corner_thresh_deg:
                is_max = True
                for d in range(-k, k + 1):
                    if d == 0:
                        continue
                    neighbor_idx = (i + d) % n if is_closed else (i + d)
                    if 0 <= neighbor_idx < n:
                        if angles[neighbor_idx] > angles[i]:
                            is_max = False
                            break
                if is_max:
                    corners.append(i)

        return corners

    @staticmethod
    def is_collinear(pts: np.ndarray, tolerance: float = 0.65) -> bool:
        """
        Tests if intermediate points in pts lie within tolerance pixels of the straight chord
        connecting pts[0] and pts[-1].
        """
        n = len(pts)
        if n <= 2:
            return True

        p0 = pts[0]
        p3 = pts[-1]
        v = p3 - p0
        line_len = np.hypot(v[0], v[1])
        if line_len < 1e-4:
            # Points are all coincident at same location
            return True

        # Distance of each point from line: |v_x * (y0 - y) - v_y * (x0 - x)| / line_len
        cross = np.abs(v[0] * (p0[1] - pts[:, 1]) - v[1] * (p0[0] - pts[:, 0]))
        dists = cross / line_len
        return bool(np.max(dists) <= tolerance)


class AdaptiveBezierFitter:
    """
    Fits high-precision Cubic Bézier curve equations (and collinear 'L' segments)
    to 2D coordinate chains with sharp corner preservation.
    """

    def __init__(
        self,
        max_error: float = 1.0,
        corner_thresh_deg: float = 50.0,
        collinear_thresh: float = 0.65,
        precision: int = 0,
    ):
        self.max_error = max_error
        self.corner_thresh_deg = corner_thresh_deg
        self.collinear_thresh = collinear_thresh
        self.precision = precision
        self.analyzer = CurvatureAnalyzer()

    def _fmt(self, val: float) -> str:
        if self.precision <= 0:
            return str(int(round(val)))
        rounded = round(val, self.precision)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.{self.precision}f}"

    def _estimate_tangent(
        self, pts: np.ndarray, idx: int, forward: bool, span: int = 2
    ) -> np.ndarray:
        """Estimates normalized tangent vector at an endpoint or split point."""
        n = len(pts)
        if forward:
            end = min(n - 1, idx + span)
            vec = pts[end] - pts[idx]
        else:
            start = max(0, idx - span)
            vec = pts[idx] - pts[start]

        dist = np.hypot(vec[0], vec[1])
        if dist < 1e-4:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (vec / dist).astype(np.float32)

    def _fit_single_cubic(
        self, pts: np.ndarray, t1: np.ndarray, t2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fits a single cubic Bézier to pts with given end tangents t1 and t2 using least squares.
        Returns: (P1, P2, max_error).
        """
        p0 = pts[0]
        p3 = pts[-1]
        chord_len = float(np.hypot(p3[0] - p0[0], p3[1] - p0[1]))

        # Chord-length parameterization
        diffs = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        u = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = u[-1]
        if total_len < 1e-4:
            return p0.copy(), p3.copy(), 0.0
        u /= total_len

        # Bernstein basis polynomials
        b0 = (1.0 - u) ** 3
        b1 = 3.0 * (1.0 - u) ** 2 * u
        b2 = 3.0 * (1.0 - u) * u ** 2
        b3 = u ** 3

        # Formulate 2x2 normal equations: B(u) = b0*P0 + b1*(P0+a1*t1) + b2*(P3-a2*t2) + b3*P3
        a1 = b1[:, None] * t1[None, :]
        a2 = -b2[:, None] * t2[None, :]
        rhs = pts - ((b0 + b1)[:, None] * p0[None, :] + (b2 + b3)[:, None] * p3[None, :])

        c11 = float(np.sum(np.sum(a1 * a1, axis=1)))
        c12 = float(np.sum(np.sum(a1 * a2, axis=1)))
        c22 = float(np.sum(np.sum(a2 * a2, axis=1)))

        x1 = float(np.sum(np.sum(a1 * rhs, axis=1)))
        x2 = float(np.sum(np.sum(a2 * rhs, axis=1)))

        det = c11 * c22 - c12 * c12

        if abs(det) > 1e-6:
            alpha1 = (x1 * c22 - x2 * c12) / det
            alpha2 = (c11 * x2 - c12 * x1) / det
            if alpha1 <= 0.0 or alpha2 <= 0.0:
                alpha1 = alpha2 = chord_len / 3.0
        else:
            alpha1 = alpha2 = chord_len / 3.0

        p1 = p0 + alpha1 * t1
        p2 = p3 - alpha2 * t2

        # Measure maximum deviation from original points
        curve = (
            b0[:, None] * p0[None, :]
            + b1[:, None] * p1[None, :]
            + b2[:, None] * p2[None, :]
            + b3[:, None] * p3[None, :]
        )
        errors = np.hypot(curve[:, 0] - pts[:, 0], curve[:, 1] - pts[:, 1])
        max_err = float(np.max(errors))
        return p1, p2, max_err

    def _fit_recursive(
        self,
        pts: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        cmds: List[str],
        depth: int = 0,
    ) -> None:
        """Recursively fits cubic Bézier segments or line segments to pts."""
        n = len(pts)
        if n < 2:
            return

        p3 = pts[-1]

        # 1. Collinear check: if all points lie within collinear_thresh of chord line, emit 'L'
        if self.analyzer.is_collinear(pts, tolerance=self.collinear_thresh):
            cmds.append(f"L{self._fmt(p3[0])} {self._fmt(p3[1])}")
            return

        # 2. Try single cubic Bézier fit
        p1, p2, max_err = self._fit_single_cubic(pts, t1, t2)
        if max_err <= self.max_error or depth >= 4 or n <= 4:
            cmds.append(
                f"C{self._fmt(p1[0])} {self._fmt(p1[1])} "
                f"{self._fmt(p2[0])} {self._fmt(p2[1])} "
                f"{self._fmt(p3[0])} {self._fmt(p3[1])}"
            )
            return

        # 3. Subdivide at point of maximum deviation
        split_idx = n // 2
        # Center tangent
        t_center = self._estimate_tangent(pts, split_idx, forward=True)

        self._fit_recursive(pts[: split_idx + 1], t1, t_center, cmds, depth + 1)
        self._fit_recursive(pts[split_idx:], t_center, t2, cmds, depth + 1)

    def fit_stroke_to_path(
        self,
        points: np.ndarray,
        is_closed: bool = False,
    ) -> str:
        """
        Converts a sequence of 2D points into an optimized SVG path string (M, L, C, Z)
        with sharp corner preservation and collinear reduction.
        """
        n = len(points)
        if n < 2:
            return ""

        if n <= 3:
            # Small segments: straight lines
            coords = "L".join(f"{self._fmt(p[0])} {self._fmt(p[1])}" for p in points[1:])
            suffix = "Z" if is_closed else ""
            return f"M{self._fmt(points[0][0])} {self._fmt(points[0][1])}L{coords}{suffix}"

        # 1. Detect sharp corners along the stroke
        corners = self.analyzer.detect_corners(
            points, is_closed=is_closed, corner_thresh_deg=self.corner_thresh_deg
        )

        cmds: List[str] = [f"M{self._fmt(points[0][0])} {self._fmt(points[0][1])}"]

        # 2. Split stroke into smooth sub-segments between sharp corners
        if not corners:
            # Entire stroke is smooth
            t1 = self._estimate_tangent(points, 0, forward=True)
            t2 = self._estimate_tangent(points, n - 1, forward=False)
            self._fit_recursive(points, t1, t2, cmds)
        else:
            # Partition into slices at corner indices
            split_indices = [0] + [c for c in corners if 0 < c < n - 1] + [n - 1]
            # Remove consecutive duplicates
            split_indices = sorted(list(set(split_indices)))

            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i + 1]
                slice_pts = points[start : end + 1]

                t1 = self._estimate_tangent(slice_pts, 0, forward=True)
                t2 = self._estimate_tangent(slice_pts, len(slice_pts) - 1, forward=False)
                self._fit_recursive(slice_pts, t1, t2, cmds)

        if is_closed:
            cmds.append("Z")

        return "".join(cmds)
