"""Recommendation service for garment-body fit analysis and preference scoring.

This service combines:
1. Body measurements (b_spec) from SMPL worker stored in MinIO
2. Garment specifications (g_spec) from local catalog
3. User preferences from preference model (based on likes/ratings)

The recommendation flow:
1. Load body measurements from MinIO (smpl-measurements bucket)
2. Load garment catalog from local JSON file
3. Apply fit-based filtering using FitRecommender
4. Apply preference scoring using PreferenceModel
5. Combine scores and return ranked recommendations
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from closet_canvas.recommendation.recommender import (
    BodyMeasurements,
    FitAnalysis,
    FitRecommender,
    GarmentMeasurements,
)
from closet_canvas.recommendation.preference import (
    PreferenceConfig,
    PreferenceModel,
)


@dataclass
class GarmentCatalogItem:
    """Full garment catalog item with measurements and metadata."""

    garment_id: str
    title: str
    brand: str
    category: str
    size: str
    price: float
    image_url: str
    chest_cm: float
    shoulder_cm: float
    waist_cm: float
    hip_cm: float
    embedding: Optional[np.ndarray] = None

    def to_garment_measurements(self) -> GarmentMeasurements:
        """Convert to GarmentMeasurements for fit analysis."""
        return GarmentMeasurements.from_json({
            "garment_id": self.garment_id,
            "size": self.size,
            "chest_cm": self.chest_cm,
            "shoulder_cm": self.shoulder_cm,
            "waist_cm": self.waist_cm,
            "hip_cm": self.hip_cm,
            "category": self.category,
            "brand": self.brand,
        })

    def to_recommendation_dict(
        self,
        fit_score: float,
        preference_score: float,
        combined_score: float,
    ) -> Dict[str, Any]:
        """Convert to recommendation response format."""
        return {
            "id": self.garment_id,
            "imageUrl": self.image_url,
            "title": self.title,
            "brand": self.brand,
            "price": self.price,
            "category": self.category,
            "matchScore": combined_score,
            "fitScore": fit_score,
            "preferenceScore": preference_score,
            "size": self.size,
        }


class GarmentCatalogService:
    """Service for loading and managing garment catalog."""

    def __init__(self, catalog_path: Optional[str] = None):
        """Initialize catalog service.

        Args:
            catalog_path: Path to catalog JSON file. If None, uses default location.
        """
        self._catalog_path = catalog_path or self._get_default_catalog_path()
        self._catalog: List[GarmentCatalogItem] = []
        self._loaded = False

    def _get_default_catalog_path(self) -> str:
        """Get default catalog path based on environment."""
        # Check for mounted catalog in Docker
        docker_path = "/app/data/catalog/garments.json"
        if os.path.exists(docker_path):
            return docker_path

        # Check for local development path
        local_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "data", "catalog", "garments.json"
        )
        if os.path.exists(local_path):
            return local_path

        # Default to Docker path (will be created)
        return docker_path

    def load_catalog(self) -> List[GarmentCatalogItem]:
        """Load garment catalog from JSON file."""
        if self._loaded:
            return self._catalog

        try:
            with open(self._catalog_path, "r") as f:
                data = json.load(f)

            self._catalog = []
            for item in data.get("garments", []):
                catalog_item = GarmentCatalogItem(
                    garment_id=item["garment_id"],
                    title=item["title"],
                    brand=item["brand"],
                    category=item["category"],
                    size=item["size"],
                    price=item["price"],
                    image_url=item["image_url"],
                    chest_cm=item["measurements"]["chest_cm"],
                    shoulder_cm=item["measurements"]["shoulder_cm"],
                    waist_cm=item["measurements"]["waist_cm"],
                    hip_cm=item["measurements"]["hip_cm"],
                    embedding=np.array(item["embedding"]) if item.get("embedding") else None,
                )
                self._catalog.append(catalog_item)

            self._loaded = True
            print(f"[RecommendationService] Loaded {len(self._catalog)} garments from catalog")
            return self._catalog

        except FileNotFoundError:
            print(f"[RecommendationService] Catalog not found at {self._catalog_path}, using sample data")
            self._catalog = self._create_sample_catalog()
            self._loaded = True
            return self._catalog

        except Exception as e:
            print(f"[RecommendationService] Error loading catalog: {e}")
            self._catalog = self._create_sample_catalog()
            self._loaded = True
            return self._catalog

    def _create_sample_catalog(self) -> List[GarmentCatalogItem]:
        """Create sample catalog for development/testing."""
        sample_items = [
            GarmentCatalogItem(
                garment_id="G001",
                title="Classic Cotton T-Shirt",
                brand="Nike",
                category="tops",
                size="M",
                price=35.00,
                image_url="https://picsum.photos/seed/nike1/400/500",
                chest_cm=96.0,
                shoulder_cm=45.5,
                waist_cm=83.0,
                hip_cm=99.0,
            ),
            GarmentCatalogItem(
                garment_id="G002",
                title="Slim Fit Polo",
                brand="Adidas",
                category="tops",
                size="L",
                price=45.00,
                image_url="https://picsum.photos/seed/adidas1/400/550",
                chest_cm=100.0,
                shoulder_cm=47.0,
                waist_cm=87.0,
                hip_cm=103.0,
            ),
            GarmentCatalogItem(
                garment_id="G003",
                title="Performance Tank",
                brand="Puma",
                category="tops",
                size="S",
                price=28.00,
                image_url="https://picsum.photos/seed/puma1/400/480",
                chest_cm=92.0,
                shoulder_cm=43.0,
                waist_cm=79.0,
                hip_cm=95.0,
            ),
            GarmentCatalogItem(
                garment_id="G004",
                title="Casual Blazer",
                brand="Zara",
                category="outerwear",
                size="M",
                price=129.00,
                image_url="https://picsum.photos/seed/zara1/400/600",
                chest_cm=98.0,
                shoulder_cm=46.0,
                waist_cm=85.0,
                hip_cm=100.0,
            ),
            GarmentCatalogItem(
                garment_id="G005",
                title="Denim Jacket",
                brand="Levi's",
                category="outerwear",
                size="M",
                price=89.00,
                image_url="https://picsum.photos/seed/levis1/400/520",
                chest_cm=97.0,
                shoulder_cm=45.0,
                waist_cm=84.0,
                hip_cm=98.0,
            ),
            GarmentCatalogItem(
                garment_id="G006",
                title="Slim Chinos",
                brand="H&M",
                category="bottoms",
                size="32",
                price=39.00,
                image_url="https://picsum.photos/seed/hm1/400/580",
                chest_cm=0.0,  # Not applicable for bottoms
                shoulder_cm=0.0,
                waist_cm=82.0,
                hip_cm=96.0,
            ),
            GarmentCatalogItem(
                garment_id="G007",
                title="Athletic Joggers",
                brand="Under Armour",
                category="bottoms",
                size="M",
                price=55.00,
                image_url="https://picsum.photos/seed/ua1/400/560",
                chest_cm=0.0,
                shoulder_cm=0.0,
                waist_cm=80.0,
                hip_cm=98.0,
            ),
            GarmentCatalogItem(
                garment_id="G008",
                title="Formal Shirt",
                brand="Ralph Lauren",
                category="tops",
                size="M",
                price=95.00,
                image_url="https://picsum.photos/seed/rl1/400/540",
                chest_cm=95.0,
                shoulder_cm=44.5,
                waist_cm=82.0,
                hip_cm=97.0,
            ),
            GarmentCatalogItem(
                garment_id="G009",
                title="Hoodie",
                brand="Champion",
                category="tops",
                size="L",
                price=65.00,
                image_url="https://picsum.photos/seed/champ1/400/530",
                chest_cm=102.0,
                shoulder_cm=48.0,
                waist_cm=90.0,
                hip_cm=105.0,
            ),
            GarmentCatalogItem(
                garment_id="G010",
                title="Bomber Jacket",
                brand="Alpha Industries",
                category="outerwear",
                size="M",
                price=175.00,
                image_url="https://picsum.photos/seed/alpha1/400/550",
                chest_cm=99.0,
                shoulder_cm=46.5,
                waist_cm=86.0,
                hip_cm=101.0,
            ),
            GarmentCatalogItem(
                garment_id="G011",
                title="V-Neck Sweater",
                brand="Gap",
                category="tops",
                size="M",
                price=49.00,
                image_url="https://picsum.photos/seed/gap1/400/510",
                chest_cm=96.0,
                shoulder_cm=45.0,
                waist_cm=84.0,
                hip_cm=98.0,
            ),
            GarmentCatalogItem(
                garment_id="G012",
                title="Cargo Pants",
                brand="Carhartt",
                category="bottoms",
                size="32",
                price=75.00,
                image_url="https://picsum.photos/seed/carhartt1/400/590",
                chest_cm=0.0,
                shoulder_cm=0.0,
                waist_cm=83.0,
                hip_cm=99.0,
            ),
        ]
        return sample_items

    def get_all_garments(self) -> List[GarmentCatalogItem]:
        """Get all garments from catalog."""
        if not self._loaded:
            self.load_catalog()
        return self._catalog

    def get_garments_by_category(self, category: str) -> List[GarmentCatalogItem]:
        """Get garments filtered by category."""
        return [g for g in self.get_all_garments() if g.category == category]

    def get_garment_by_id(self, garment_id: str) -> Optional[GarmentCatalogItem]:
        """Get a specific garment by ID."""
        for g in self.get_all_garments():
            if g.garment_id == garment_id:
                return g
        return None


class RecommendationService:
    """Service for generating personalized garment recommendations."""

    def __init__(
        self,
        storage_gateway,
        catalog_service: Optional[GarmentCatalogService] = None,
        fit_weight: float = 0.6,
        preference_weight: float = 0.4,
    ):
        """Initialize recommendation service.

        Args:
            storage_gateway: Storage gateway for MinIO access
            catalog_service: Optional catalog service (creates default if None)
            fit_weight: Weight for fit score in combined ranking (0-1)
            preference_weight: Weight for preference score in combined ranking (0-1)
        """
        self._storage = storage_gateway
        self._catalog = catalog_service or GarmentCatalogService()
        self._fit_weight = fit_weight
        self._preference_weight = preference_weight
        self._recommender = FitRecommender()

    def load_body_measurements(
        self,
        session_id: str,
        clip_id: str,
    ) -> Optional[BodyMeasurements]:
        """Load body measurements from MinIO storage.

        Args:
            session_id: User's capture session ID
            clip_id: Specific clip ID

        Returns:
            BodyMeasurements object or None if not found
        """
        bucket = "smpl-measurements"
        key = f"{session_id}/{clip_id}/measurements.json"

        try:
            # Download measurements JSON
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name

            self._storage.download(bucket, key, tmp_path)

            with open(tmp_path, "r") as f:
                data = json.load(f)

            os.unlink(tmp_path)

            # Add session/clip info
            data["session_id"] = session_id
            data["clip_id"] = clip_id

            return BodyMeasurements.from_json(data)

        except Exception as e:
            print(f"[RecommendationService] Failed to load measurements: {e}")
            return None

    def load_user_preferences(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[PreferenceModel]:
        """Load user preference model from storage.

        Args:
            session_id: Current session ID
            user_id: Optional user ID for persistent preferences

        Returns:
            PreferenceModel or None if not found
        """
        bucket = "user-preferences"
        key = f"{user_id or session_id}/preference_model.json"

        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name

            self._storage.download(bucket, key, tmp_path)

            with open(tmp_path, "r") as f:
                data = f.read()

            os.unlink(tmp_path)

            return PreferenceModel.deserialize(data)

        except Exception as e:
            print(f"[RecommendationService] No preference model found: {e}")
            return None

    def generate_recommendations(
        self,
        session_id: str,
        clip_id: str,
        user_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        min_fit_score: float = 25.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations for a session.

        Args:
            session_id: User's capture session ID
            clip_id: Specific clip ID with body measurements
            user_id: Optional user ID for preferences
            categories: Optional list of categories to filter by
            min_fit_score: Minimum fit score threshold (0-100)
            limit: Maximum number of recommendations to return

        Returns:
            List of recommendation dictionaries sorted by combined score
        """
        # Load body measurements
        body_measurements = self.load_body_measurements(session_id, clip_id)
        if body_measurements is None:
            print(f"[RecommendationService] No body measurements for {session_id}/{clip_id}")
            return []

        print(f"[RecommendationService] Body measurements loaded:")
        print(f"  Chest: {body_measurements.chest}cm")
        print(f"  Shoulder: {body_measurements.shoulder}cm")
        print(f"  Waist: {body_measurements.waist}cm")
        print(f"  Hip: {body_measurements.hip}cm")

        # Load garment catalog
        garments = self._catalog.get_all_garments()
        if categories:
            garments = [g for g in garments if g.category in categories]

        if not garments:
            print("[RecommendationService] No garments in catalog")
            return []

        print(f"[RecommendationService] Analyzing {len(garments)} garments")

        # Load user preferences (optional)
        preference_model = self.load_user_preferences(session_id, user_id)

        # Score each garment
        recommendations = []
        for garment in garments:
            # Skip bottoms that don't have upper body measurements
            if garment.category == "bottoms":
                # For bottoms, only check waist and hip
                fit_score = self._calculate_bottoms_fit_score(
                    body_measurements, garment
                )
            else:
                # Full fit analysis for tops/outerwear
                garment_measurements = garment.to_garment_measurements()
                analysis = self._recommender.analyze_fit(
                    body_measurements, garment_measurements
                )
                fit_score = analysis.fit_score

            # Skip if below minimum fit score
            if fit_score < min_fit_score:
                continue

            # Calculate preference score (0-100)
            preference_score = 50.0  # Default neutral
            if preference_model and garment.embedding is not None:
                # Score ranges from -1 to 1, normalize to 0-100
                raw_score = preference_model.score(garment.embedding)
                preference_score = (raw_score + 1.0) * 50.0

            # Calculate combined score
            combined_score = (
                self._fit_weight * (fit_score / 100.0)
                + self._preference_weight * (preference_score / 100.0)
            )

            recommendations.append(
                garment.to_recommendation_dict(
                    fit_score=fit_score,
                    preference_score=preference_score,
                    combined_score=combined_score,
                )
            )

        # Sort by combined score descending
        recommendations.sort(key=lambda x: x["matchScore"], reverse=True)

        # Limit results
        recommendations = recommendations[:limit]

        print(f"[RecommendationService] Generated {len(recommendations)} recommendations")
        return recommendations

    def _calculate_bottoms_fit_score(
        self,
        body: BodyMeasurements,
        garment: GarmentCatalogItem,
    ) -> float:
        """Calculate fit score for bottoms (only waist and hip)."""
        score = 0.0

        # Waist comparison
        waist_diff = abs(garment.waist_cm - body.waist)
        if 1.0 <= waist_diff <= 3.0:
            score += 50.0

        # Hip comparison
        hip_diff = abs(garment.hip_cm - body.hip)
        if 1.0 <= hip_diff <= 3.0:
            score += 50.0

        return score

    def update_user_preference(
        self,
        session_id: str,
        user_id: Optional[str],
        garment_id: str,
        rating: int,
    ) -> bool:
        """Update user preference based on garment rating.

        Args:
            session_id: Current session ID
            user_id: Optional user ID
            garment_id: ID of rated garment
            rating: User rating (1-5)

        Returns:
            True if preference updated successfully
        """
        try:
            # Load existing preference model or create new
            preference_model = self.load_user_preferences(session_id, user_id)
            if preference_model is None:
                # Create new model with default dimension (512 for CLIP)
                config = PreferenceConfig(dimension=512, learning_rate=0.1)
                preference_model = PreferenceModel(config)

            # Get garment embedding
            garment = self._catalog.get_garment_by_id(garment_id)
            if garment is None or garment.embedding is None:
                print(f"[RecommendationService] Garment {garment_id} not found or no embedding")
                return False

            # Update preference
            preference_model.update(garment.embedding, rating)

            # Save updated preference model
            bucket = "user-preferences"
            key = f"{user_id or session_id}/preference_model.json"

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                tmp.write(preference_model.serialize())
                tmp_path = tmp.name

            self._storage.upload(bucket, key, tmp_path)
            os.unlink(tmp_path)

            print(f"[RecommendationService] Updated preference for {user_id or session_id}")
            return True

        except Exception as e:
            print(f"[RecommendationService] Failed to update preference: {e}")
            return False


def create_recommendation_service(config) -> RecommendationService:
    """Factory function to create recommendation service with dependencies."""
    from .storage import create_storage_gateway

    storage = create_storage_gateway(config)
    catalog = GarmentCatalogService()

    return RecommendationService(
        storage_gateway=storage,
        catalog_service=catalog,
    )
