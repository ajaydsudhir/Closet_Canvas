"""Use cases for ingest service."""

from .gate_clip import GateClipUseCase
from .process_clip import ProcessClipUseCase

# Recommendation use cases are imported lazily to avoid numpy dependency in CPU worker
# from .recommend import GenerateRecommendationsUseCase, UpdatePreferenceUseCase

__all__ = [
    "GateClipUseCase",
    "ProcessClipUseCase",
]
