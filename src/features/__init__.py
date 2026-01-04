"""Módulo de extracción de características."""

from .pipeline import FeaturePipeline
from .optical_flow import OpticalFlowExtractor
from .temporal import TemporalFeatureExtractor
from .spatial import SpatialFeatureExtractor

__all__ = [
    'FeaturePipeline',
    'OpticalFlowExtractor', 
    'TemporalFeatureExtractor',
    'SpatialFeatureExtractor'
]
