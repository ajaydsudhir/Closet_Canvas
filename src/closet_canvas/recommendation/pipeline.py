"""
Integrated Recommendation Pipeline for Closet Canvas.

This module combines:
1. Preference scoring (style matching via embeddings)
2. Fit validation (body vs garment measurements)
3. Top-N ranking and result formatting

Output: Top 5 garments with MinIO image URLs for frontend display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import numpy as np

from .preference import PreferenceModel
from .recommender import (
    BodyMeasurements,
    GarmentMeasurements,
    FitRecommender,
    FitAnalysis,
)


@dataclass(frozen=True)
class GarmentCandidate:
    """Complete garment information including measurements and embeddings."""

    garment_id: str
    image_url: str  # MinIO URL
    title: str
    brand: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    
    # Measurements
    chest_cm: float = 0.0
    shoulder_cm: float = 0.0
    waist_cm: float = 0.0
    hip_cm: float = 0.0
    size: Optional[str] = None
    
    # Style embedding (flattened to 1D)
    embedding: Optional[np.ndarray] = None
    
    @classmethod
    def from_json(cls, data: Dict) -> "GarmentCandidate":
        """Parse from database/MinIO JSON."""
        embedding = data.get("embedding")
        if embedding is not None:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)
        
        return cls(
            garment_id=str(data.get("garment_id", "")),
            image_url=str(data.get("image_url", "")),
            title=str(data.get("title", "Untitled Garment")),
            brand=data.get("brand"),
            price=data.get("price"),
            category=data.get("category"),
            chest_cm=float(data.get("chest_cm", 0)),
            shoulder_cm=float(data.get("shoulder_cm", 0)),
            waist_cm=float(data.get("waist_cm", 0)),
            hip_cm=float(data.get("hip_cm", 0)),
            size=data.get("size"),
            embedding=embedding,
        )
    
    def to_garment_measurements(self) -> GarmentMeasurements:
        """Convert to GarmentMeasurements for fit analysis."""
        return GarmentMeasurements(
            chest=self.chest_cm,
            shoulder=self.shoulder_cm,
            waist=self.waist_cm,
            hip=self.hip_cm,
            garment_id=self.garment_id,
            size=self.size,
            category=self.category,
            brand=self.brand,
        )


@dataclass(frozen=True)
class RecommendationResult:
    """Final recommendation with all scores and metadata."""

    garment_id: str
    image_url: str
    title: str
    brand: Optional[str]
    price: Optional[float]
    category: Optional[str]
    
    # Scores
    preference_score: float  # 0-1, cosine similarity
    fit: int  # 1 = fits, 0 = doesn't fit
    fit_score: float  # 0-100
    combined_score: float  # Overall ranking score
    
    # Metadata
    size: Optional[str]
    recommendation_text: str
    
    def to_frontend_dict(self) -> Dict:
        """Convert to frontend RecommendationItem format."""
        return {
            "id": self.garment_id,
            "image_url": self.image_url,  # MinIO URL
            "title": self.title,
            "brand": self.brand,
            "price": self.price,
            "category": self.category,
            "match_score": self.combined_score,  # 0-1 for frontend display
        }


class RecommendationPipeline:
    """
    Integrated pipeline: preference → fit → ranking → top-N.
    
    Workflow:
    1. Score all candidates by preference (style)
    2. Filter to candidates with embeddings
    3. Validate fit for top-K preference matches
    4. Combine scores and return top-N
    """
    
    def __init__(
        self,
        preference_model: PreferenceModel,
        fit_recommender: Optional[FitRecommender] = None,
        preference_weight: float = 0.4,
        fit_weight: float = 0.6,
    ):
        """
        Initialize pipeline with models and weights.
        
        Args:
            preference_model: Trained preference model
            fit_recommender: Fit validation model (uses default if None)
            preference_weight: Weight for preference score (0-1)
            fit_weight: Weight for fit score (0-1)
        """
        self.preference_model = preference_model
        self.fit_recommender = fit_recommender or FitRecommender()
        self.preference_weight = preference_weight
        self.fit_weight = fit_weight
    
    def score_candidates(
        self,
        candidates: List[GarmentCandidate],
        body_measurements: BodyMeasurements,
        top_n: int = 5,
        require_fit: bool = True,
    ) -> List[RecommendationResult]:
        """
        Score and rank candidates, returning top-N recommendations.
        
        Args:
            candidates: List of garment candidates with embeddings
            body_measurements: User's body measurements
            top_n: Number of recommendations to return
            require_fit: If True, only return garments that fit (fit=1)
        
        Returns:
            List of RecommendationResult sorted by combined_score
        """
        results = []
        
        for candidate in candidates:
            # Skip if no embedding
            if candidate.embedding is None:
                continue
            
            # 1. Preference score (style matching)
            try:
                preference_score = self.preference_model.score(candidate.embedding)
                # Normalize to 0-1 (cosine similarity is -1 to 1)
                preference_score = (preference_score + 1.0) / 2.0
            except Exception as e:
                print(f"[Pipeline] Error scoring {candidate.garment_id}: {e}")
                preference_score = 0.5  # neutral
            
            # 2. Fit validation
            garment_meas = candidate.to_garment_measurements()
            fit_analysis = self.fit_recommender.analyze_fit(
                body_measurements, garment_meas
            )
            
            # Skip if fit required but doesn't fit
            if require_fit and fit_analysis.overall_fit.value == 0:
                continue
            
            # 3. Combined score
            # Normalize fit_score (0-100) to 0-1
            fit_score_norm = fit_analysis.fit_score / 100.0
            combined_score = (
                self.preference_weight * preference_score +
                self.fit_weight * fit_score_norm
            )
            
            result = RecommendationResult(
                garment_id=candidate.garment_id,
                image_url=candidate.image_url,
                title=candidate.title,
                brand=candidate.brand,
                price=candidate.price,
                category=candidate.category,
                preference_score=preference_score,
                fit=fit_analysis.overall_fit.value,
                fit_score=fit_analysis.fit_score,
                combined_score=combined_score,
                size=candidate.size,
                recommendation_text=fit_analysis.recommendation,
            )
            results.append(result)
        
        # Sort by combined score (descending)
        results.sort(key=lambda r: r.combined_score, reverse=True)
        
        # Return top-N
        return results[:top_n]
    
    def generate_recommendations(
        self,
        candidates: List[GarmentCandidate],
        body_measurements: BodyMeasurements,
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Full pipeline returning frontend-ready JSON.
        
        Returns:
            List of dicts matching frontend RecommendationItem type
        """
        results = self.score_candidates(
            candidates, body_measurements, top_n=top_n, require_fit=True
        )
        
        # If we don't have enough fitting items, relax fit requirement
        if len(results) < top_n:
            print(f"[Pipeline] Only {len(results)} items fit, relaxing requirement...")
            results = self.score_candidates(
                candidates, body_measurements, top_n=top_n, require_fit=False
            )
        
        return [r.to_frontend_dict() for r in results]


def load_catalog_candidates_from_db(
    catalog_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
    db_connection=None,
    storage_client=None,
) -> List[GarmentCandidate]:    
    # Template data for development
    template_garments = [
        {
            "garment_id": "G001",
            "image_url": "http://localhost:9000/catalog-images/shirts/nike_001.jpg",
            "title": "Nike Performance Tee",
            "brand": "Nike",
            "price": 45.99,
            "category": "shirt",
            "chest_cm": 96.0,
            "shoulder_cm": 45.5,
            "waist_cm": 83.0,
            "hip_cm": 99.0,
            "size": "M",
            "embedding": np.random.randn(512).astype(np.float32),  # CLIP dim
        },
        {
            "garment_id": "G002",
            "image_url": "http://localhost:9000/catalog-images/shirts/adidas_002.jpg",
            "title": "Adidas Classic Fit",
            "brand": "Adidas",
            "price": 52.50,
            "category": "shirt",
            "chest_cm": 100.0,
            "shoulder_cm": 47.0,
            "waist_cm": 87.0,
            "hip_cm": 103.0,
            "size": "L",
            "embedding": np.random.randn(512).astype(np.float32),
        },
        {
            "garment_id": "G003",
            "image_url": "http://localhost:9000/catalog-images/shirts/puma_003.jpg",
            "title": "Puma Athletic Shirt",
            "brand": "Puma",
            "price": 38.00,
            "category": "shirt",
            "chest_cm": 92.0,
            "shoulder_cm": 43.0,
            "waist_cm": 79.0,
            "hip_cm": 95.0,
            "size": "S",
            "embedding": np.random.randn(512).astype(np.float32),
        },
        {
            "garment_id": "G004",
            "image_url": "http://localhost:9000/catalog-images/shirts/underarmour_004.jpg",
            "title": "Under Armour Tech Tee",
            "brand": "Under Armour",
            "price": 42.00,
            "category": "shirt",
            "chest_cm": 97.0,
            "shoulder_cm": 46.0,
            "waist_cm": 84.0,
            "hip_cm": 100.0,
            "size": "M",
            "embedding": np.random.randn(512).astype(np.float32),
        },
        {
            "garment_id": "G005",
            "image_url": "http://localhost:9000/catalog-images/shirts/reebok_005.jpg",
            "title": "Reebok Training Top",
            "brand": "Reebok",
            "price": 35.99,
            "category": "shirt",
            "chest_cm": 94.0,
            "shoulder_cm": 44.5,
            "waist_cm": 81.0,
            "hip_cm": 97.0,
            "size": "M",
            "embedding": np.random.randn(512).astype(np.float32),
        },
        {
            "garment_id": "G006",
            "image_url": "http://localhost:9000/catalog-images/shirts/champion_006.jpg",
            "title": "Champion Crew Neck",
            "brand": "Champion",
            "price": 29.99,
            "category": "shirt",
            "chest_cm": 98.0,
            "shoulder_cm": 46.5,
            "waist_cm": 85.0,
            "hip_cm": 101.0,
            "size": "M",
            "embedding": np.random.randn(512).astype(np.float32),
        },
        {
            "garment_id": "G007",
            "image_url": "http://localhost:9000/catalog-images/shirts/newbalance_007.jpg",
            "title": "New Balance Essential Tee",
            "brand": "New Balance",
            "price": 32.50,
            "category": "shirt",
            "chest_cm": 95.0,
            "shoulder_cm": 45.0,
            "waist_cm": 82.0,
            "hip_cm": 98.0,
            "size": "M",
            "embedding": np.random.randn(512).astype(np.float32),
        },
    ]
    
    return [GarmentCandidate.from_json(g) for g in template_garments]


def generate_recommendations_for_session(
    session_id: str,
    clip_id: str,
    preference_model: PreferenceModel,
    storage_client=None,
    db_connection=None,
    top_n: int = 5,
) -> List[Dict]:
    """
    Complete pipeline for generating recommendations for a session.
    
    Args:
        session_id: User's capture session ID
        clip_id: Specific clip ID
        preference_model: Trained preference model
        storage_client: MinIO/S3 client
        db_connection: Database connection
        top_n: Number of recommendations to return
    
    Returns:
        List of recommendation dicts ready for frontend
    """
    # Import here to avoid circular dependency
    from .recommender import load_body_measurements_from_minio
    
    # 1. Load body measurements
    body_measurements = load_body_measurements_from_minio(
        session_id, clip_id, storage_client
    )
    
    if body_measurements is None:
        print(f"[Pipeline] No body measurements found for {session_id}/{clip_id}")
        return []
    
    # 2. Load catalog candidates
    candidates = load_catalog_candidates_from_db(
        db_connection=db_connection,
        storage_client=storage_client,
    )
    
    if not candidates:
        print("[Pipeline] No catalog candidates found")
        return []
    
    # 3. Run pipeline
    pipeline = RecommendationPipeline(preference_model)
    recommendations = pipeline.generate_recommendations(
        candidates, body_measurements, top_n=top_n
    )
    
    return recommendations


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from .preference import PreferenceConfig, PreferenceModel
    
    # 1. Create preference model
    cfg = PreferenceConfig(dimension=512)  # CLIP dimension
    pref_model = PreferenceModel(cfg, seed=42)
    
    # 2. Simulate user ratings (would come from UI)
    sample_embeddings = [np.random.randn(512).astype(np.float32) for _ in range(3)]
    sample_ratings = [5, 4, 2]  # like, like, dislike
    pref_model.bulk_update(sample_embeddings, sample_ratings)
    
    # 3. Load body measurements
    body_json = {
        "chest_circumference_cm": 95.0,
        "shoulder_width_cm": 45.0,
        "waist_circumference_cm": 82.0,
        "hip_circumference_cm": 98.0,
        "height_cm": 175.0,
    }
    body = BodyMeasurements.from_json(body_json)
    
    # 4. Load candidates
    candidates = load_catalog_candidates_from_db()
    
    # 5. Generate recommendations
    pipeline = RecommendationPipeline(pref_model)
    recommendations = pipeline.generate_recommendations(candidates, body, top_n=5)
    
    print(f"\n{'='*70}")
    print("TOP 5 RECOMMENDATIONS")
    print(f"{'='*70}")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']} by {rec['brand']}")
        print(f"   ID: {rec['id']}")
        print(f"   Image: {rec['image_url']}")
        print(f"   Match Score: {rec['match_score']:.2%}")
        print(f"   Price: ${rec['price']:.2f}" if rec['price'] else "")
