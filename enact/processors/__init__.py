"""
Processors module for ENACT

Contains batch processors for running core algorithms on multiple tasks.
"""

from enact.processors.segmentation_processor import SegmentationProcessor
from enact.processors.evaluator_processor import EvaluatorProcessor

__all__ = [
    'SegmentationProcessor',
    'EvaluatorProcessor',
]
