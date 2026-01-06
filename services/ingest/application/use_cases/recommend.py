"""Use case for generating garment recommendations.

This use case orchestrates the recommendation generation process:
1. Loads body measurements from SMPL processing
2. Loads garment catalog
3. Applies fit-based filtering
4. Applies preference scoring
5. Returns ranked recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..interfaces import StorageGateway
from services.ingest.infrastructure.recommendation_service import (
    GarmentCatalogService,
    RecommendationService,
)


@dataclass
class RecommendationJob:
    """Job data for recommendation processing."""

    session_id: str
    clip_id: str
    user_id: Optional[str] = None
    categories: Optional[List[str]] = None
    min_fit_score: float = 25.0
    limit: int = 20


@dataclass
class RecommendationResult:
    """Result of recommendation processing."""

    session_id: str
    clip_id: str
    recommendations: List[Dict[str, Any]]
    total_count: int
    success: bool
    error: Optional[str] = None


class GenerateRecommendationsUseCase:
    """Use case for generating personalized garment recommendations."""

    def __init__(
        self,
        storage: StorageGateway,
        catalog_service: Optional[GarmentCatalogService] = None,
    ):
        """Initialize use case.

        Args:
            storage: Storage gateway for MinIO access
            catalog_service: Optional catalog service
        """
        self._service = RecommendationService(
            storage_gateway=storage,
            catalog_service=catalog_service or GarmentCatalogService(),
        )

    def execute(self, job: RecommendationJob) -> RecommendationResult:
        """Execute recommendation generation.

        Args:
            job: Recommendation job parameters

        Returns:
            RecommendationResult with ranked recommendations
        """
        try:
            recommendations = self._service.generate_recommendations(
                session_id=job.session_id,
                clip_id=job.clip_id,
                user_id=job.user_id,
                categories=job.categories,
                min_fit_score=job.min_fit_score,
                limit=job.limit,
            )

            return RecommendationResult(
                session_id=job.session_id,
                clip_id=job.clip_id,
                recommendations=recommendations,
                total_count=len(recommendations),
                success=True,
            )

        except Exception as e:
            return RecommendationResult(
                session_id=job.session_id,
                clip_id=job.clip_id,
                recommendations=[],
                total_count=0,
                success=False,
                error=str(e),
            )


@dataclass
class PreferenceUpdateJob:
    """Job data for preference update."""

    session_id: str
    user_id: Optional[str]
    garment_id: str
    rating: int  # 1-5


class UpdatePreferenceUseCase:
    """Use case for updating user preferences based on ratings."""

    def __init__(
        self,
        storage: StorageGateway,
        catalog_service: Optional[GarmentCatalogService] = None,
    ):
        """Initialize use case."""
        self._service = RecommendationService(
            storage_gateway=storage,
            catalog_service=catalog_service or GarmentCatalogService(),
        )

    def execute(self, job: PreferenceUpdateJob) -> bool:
        """Execute preference update.

        Args:
            job: Preference update job parameters

        Returns:
            True if preference updated successfully
        """
        return self._service.update_user_preference(
            session_id=job.session_id,
            user_id=job.user_id,
            garment_id=job.garment_id,
            rating=job.rating,
        )
