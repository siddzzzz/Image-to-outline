"""
Ultra-Lightweight Image-to-Outline Vector Engine.
Converts entities in images into smooth black-and-white Cubic Bézier SVG path equations.
"""

from .pipeline import ImageToOutlinePipeline, OutlineResult
from .face_enhancer import FaceEnhancer
from .flow_field import EdgeTangentFlow
from .skeletonizer import StrokeSkeletonizer, CenterlineStroke
from .vectorizer import BezierVectorizer, VectorMode

__all__ = [
    "ImageToOutlinePipeline",
    "OutlineResult",
    "FaceEnhancer",
    "EdgeTangentFlow",
    "StrokeSkeletonizer",
    "CenterlineStroke",
    "BezierVectorizer",
    "VectorMode",
]

