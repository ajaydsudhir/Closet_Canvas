"""
Fit-based Recommender Module for Closet Canvas.

This module compares body measurements (from SMPL estimation) with garment
specifications to determine fit compatibility. A garment is considered a good
fit if measurements are within 1-3 cm of the body measurements.

Architecture:
- Body measurements (b_spec) come from SMPL processing stored in MinIO
- Garment specifications (g_spec) come from catalog database
- Comparison uses tolerance-based matching for chest, shoulder, waist, and hip
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
# import json


class FitStatus(int, Enum):
    """Fit status for individual measurements."""

    ACCEPTABLE = 1  # 0-3 cm difference
    NOT_ACCEPTABLE = 0  # >3 cm difference


class OverallFitStatus(int, Enum):
    """Overall fit recommendation."""

    FIT = 1  # All measurements within 1-3 cm range
    NO_FIT = 0  # One or more measurements outside acceptable range


@dataclass(frozen=True)
class BodyMeasurements:
    """Body measurements from SMPL estimation (in cm)."""

    chest: float
    shoulder: float
    waist: float
    hip: float
    height: Optional[float] = None
    session_id: Optional[str] = None
    clip_id: Optional[str] = None

    @classmethod
    def from_json(cls, data: Dict) -> "BodyMeasurements":
        """
        Create BodyMeasurements from JSON data (e.g., from MinIO).

        Expected JSON format from SMPL worker:
        {
            "chest_circumference_cm": 95.2,
            "shoulder_width_cm": 45.8,
            "waist_circumference_cm": 82.1,
            "hip_circumference_cm": 98.5,
            "height_cm": 175.0,
            "session_id": "...",
            "clip_id": "..."
        }
        """
        return cls(
            chest=float(data.get("chest_circumference_cm", 0)),
            shoulder=float(data.get("shoulder_width_cm", 0)),
            waist=float(data.get("waist_circumference_cm", 0)),
            hip=float(data.get("hip_circumference_cm", 0)),
            height=data.get("height_cm"),
            session_id=data.get("session_id"),
            clip_id=data.get("clip_id"),
        )


@dataclass(frozen=True)
class GarmentMeasurements:
    """Garment measurements from catalog database (in cm)."""

    chest: float
    shoulder: float
    waist: float
    hip: float
    garment_id: str
    size: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None

    @classmethod
    def from_json(cls, data: Dict) -> "GarmentMeasurements":
        """
        Create GarmentMeasurements from JSON data (e.g., from database).

        Expected JSON format:
        {
            "garment_id": "G12345",
            "size": "M",
            "chest_cm": 100.0,
            "shoulder_cm": 48.0,
            "waist_cm": 85.0,
            "hip_cm": 102.0,
            "category": "shirt",
            "brand": "Nike"
        }
        """
        return cls(
            chest=float(data.get("chest_cm", 0)),
            shoulder=float(data.get("shoulder_cm", 0)),
            waist=float(data.get("waist_cm", 0)),
            hip=float(data.get("hip_cm", 0)),
            garment_id=str(data.get("garment_id", "")),
            size=data.get("size"),
            category=data.get("category"),
            brand=data.get("brand"),
        )


@dataclass(frozen=True)
class MeasurementComparison:
    """Comparison result for a single measurement."""

    body_value: float
    garment_value: float
    difference: float  # garment - body (positive = looser, negative = tighter)
    abs_difference: float
    fit_status: FitStatus

    @property
    def is_acceptable(self) -> bool:
        """Check if measurement is within acceptable range (1-3 cm difference)."""
        return self.fit_status == FitStatus.ACCEPTABLE


@dataclass(frozen=True)
class FitAnalysis:
    """Complete fit analysis for body vs garment comparison."""

    body_measurements: BodyMeasurements
    garment_measurements: GarmentMeasurements

    chest_comparison: MeasurementComparison
    shoulder_comparison: MeasurementComparison
    waist_comparison: MeasurementComparison
    hip_comparison: MeasurementComparison

    overall_fit: OverallFitStatus
    fit_score: float  # 0-100 score based on all measurements
    recommendation: str

    @property
    def is_fit(self) -> bool:
        """Check if all measurements are within acceptable range (returns 1 for fit, 0 for no fit)."""
        return self.overall_fit == OverallFitStatus.FIT

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "garment_id": self.garment_measurements.garment_id,
            "size": self.garment_measurements.size,
            "fit": self.overall_fit.value,  # 1 for fit, 0 for no fit
            "fit_score": self.fit_score,
            "recommendation": self.recommendation,
            "measurements": {
                "chest": {
                    "body": self.chest_comparison.body_value,
                    "garment": self.chest_comparison.garment_value,
                    "difference": self.chest_comparison.difference,
                    "acceptable": self.chest_comparison.fit_status.value,  # 1 or 0
                },
                "shoulder": {
                    "body": self.shoulder_comparison.body_value,
                    "garment": self.shoulder_comparison.garment_value,
                    "difference": self.shoulder_comparison.difference,
                    "acceptable": self.shoulder_comparison.fit_status.value,  # 1 or 0
                },
                "waist": {
                    "body": self.waist_comparison.body_value,
                    "garment": self.waist_comparison.garment_value,
                    "difference": self.waist_comparison.difference,
                    "acceptable": self.waist_comparison.fit_status.value,  # 1 or 0
                },
                "hip": {
                    "body": self.hip_comparison.body_value,
                    "garment": self.hip_comparison.garment_value,
                    "difference": self.hip_comparison.difference,
                    "acceptable": self.hip_comparison.fit_status.value,  # 1 or 0
                },
            },
        }


class FitRecommender:
    """
    Core recommender that compares body measurements with garment specifications.

    Binary fit recommendation (1 or 0):
    - Returns 1 (FIT) if ALL measurements are within 1-3 cm range
    - Returns 0 (NO_FIT) if ANY measurement is outside the range
    """

    def __init__(self, min_threshold: float = 1.0, max_threshold: float = 3.0):
        """
        Initialize recommender with fit thresholds.

        A garment fits if measurements are within 1-3 cm of body measurements.

        Args:
            min_threshold: Minimum difference (cm) for acceptable fit (default: 1.0)
            max_threshold: Maximum difference (cm) for acceptable fit (default: 3.0)
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def compare_measurement(
        self, body_value: float, garment_value: float
    ) -> MeasurementComparison:
        """
        Compare a single measurement between body and garment.

        Returns 1 if measurement is within 1-3 cm range, 0 otherwise.

        Args:
            body_value: Body measurement in cm
            garment_value: Garment measurement in cm

        Returns:
            MeasurementComparison with difference and fit status (1 or 0)
        """
        difference = garment_value - body_value
        abs_diff = abs(difference)

        # Check if within acceptable range (1-3 cm)
        if self.min_threshold <= abs_diff <= self.max_threshold:
            status = FitStatus.ACCEPTABLE  # 1
        else:
            status = FitStatus.NOT_ACCEPTABLE  # 0

        return MeasurementComparison(
            body_value=body_value,
            garment_value=garment_value,
            difference=difference,
            abs_difference=abs_diff,
            fit_status=status,
        )

    def calculate_fit_score(
        self,
        chest: MeasurementComparison,
        shoulder: MeasurementComparison,
        waist: MeasurementComparison,
        hip: MeasurementComparison,
    ) -> float:
        """
        Calculate overall fit score (0-100) based on all measurements.

        Each measurement that fits (within 1-3 cm) contributes 25 points.

        Args:
            chest, shoulder, waist, hip: MeasurementComparison objects

        Returns:
            Score from 0-100 (0, 25, 50, 75, or 100)
        """
        score = 0

        for comparison in [chest, shoulder, waist, hip]:
            if comparison.fit_status == FitStatus.ACCEPTABLE:
                score += 25

        return float(score)

    def determine_overall_fit(
        self,
        chest: MeasurementComparison,
        shoulder: MeasurementComparison,
        waist: MeasurementComparison,
        hip: MeasurementComparison,
    ) -> OverallFitStatus:
        """
        Determine overall fit status based on individual measurements.

        Returns 1 (FIT) if ALL measurements are within 1-3 cm range.
        Returns 0 (NO_FIT) if ANY measurement is outside the range.

        Returns:
            OverallFitStatus enum value (1 or 0)
        """
        comparisons = [chest, shoulder, waist, hip]

        # All measurements must be acceptable for overall fit
        all_acceptable = all(c.fit_status == FitStatus.ACCEPTABLE for c in comparisons)

        if all_acceptable:
            return OverallFitStatus.FIT  # 1
        else:
            return OverallFitStatus.NO_FIT  # 0

    def generate_recommendation_text(
        self,
        overall_fit: OverallFitStatus,
        chest: MeasurementComparison,
        shoulder: MeasurementComparison,
        waist: MeasurementComparison,
        hip: MeasurementComparison,
        garment: GarmentMeasurements,
    ) -> str:
        """
        Generate human-readable recommendation text.

        Returns:
            Detailed recommendation string
        """
        size_str = f"size {garment.size}" if garment.size else "this garment"

        if overall_fit == OverallFitStatus.FIT:
            return f"Recommended! {size_str.capitalize()} fits your measurements (all within 1-3 cm)."

        else:  # NO_FIT
            # Identify problem areas
            issues = []
            if chest.fit_status == FitStatus.NOT_ACCEPTABLE:
                if chest.abs_difference < self.min_threshold:
                    issues.append(f"chest too close ({chest.abs_difference:.1f}cm)")
                else:
                    if chest.difference < 0:
                        issues.append(f"chest too tight ({chest.abs_difference:.1f}cm)")
                    else:
                        issues.append(f"chest too loose ({chest.abs_difference:.1f}cm)")

            if shoulder.fit_status == FitStatus.NOT_ACCEPTABLE:
                if shoulder.abs_difference < self.min_threshold:
                    issues.append(
                        f"shoulder too close ({shoulder.abs_difference:.1f}cm)"
                    )
                else:
                    if shoulder.difference < 0:
                        issues.append(
                            f"shoulder too tight ({shoulder.abs_difference:.1f}cm)"
                        )
                    else:
                        issues.append(
                            f"shoulder too loose ({shoulder.abs_difference:.1f}cm)"
                        )

            if waist.fit_status == FitStatus.NOT_ACCEPTABLE:
                if waist.abs_difference < self.min_threshold:
                    issues.append(f"waist too close ({waist.abs_difference:.1f}cm)")
                else:
                    if waist.difference < 0:
                        issues.append(f"waist too tight ({waist.abs_difference:.1f}cm)")
                    else:
                        issues.append(f"waist too loose ({waist.abs_difference:.1f}cm)")

            if hip.fit_status == FitStatus.NOT_ACCEPTABLE:
                if hip.abs_difference < self.min_threshold:
                    issues.append(f"hip too close ({hip.abs_difference:.1f}cm)")
                else:
                    if hip.difference < 0:
                        issues.append(f"hip too tight ({hip.abs_difference:.1f}cm)")
                    else:
                        issues.append(f"hip too loose ({hip.abs_difference:.1f}cm)")

            issue_text = ", ".join(issues)
            return (
                f"Not recommended. {size_str.capitalize()} doesn't fit: {issue_text}."
            )

    def analyze_fit(
        self,
        body_spec: BodyMeasurements,
        garment_spec: GarmentMeasurements,
    ) -> FitAnalysis:
        """
        Perform complete fit analysis comparing body and garment measurements.

        Args:
            body_spec: Body measurements from SMPL estimation
            garment_spec: Garment measurements from catalog

        Returns:
            FitAnalysis with detailed comparison results
        """
        # Compare individual measurements
        chest_cmp = self.compare_measurement(body_spec.chest, garment_spec.chest)
        shoulder_cmp = self.compare_measurement(
            body_spec.shoulder, garment_spec.shoulder
        )
        waist_cmp = self.compare_measurement(body_spec.waist, garment_spec.waist)
        hip_cmp = self.compare_measurement(body_spec.hip, garment_spec.hip)

        # Calculate overall metrics
        fit_score = self.calculate_fit_score(
            chest_cmp, shoulder_cmp, waist_cmp, hip_cmp
        )
        overall_fit = self.determine_overall_fit(
            chest_cmp, shoulder_cmp, waist_cmp, hip_cmp
        )
        recommendation = self.generate_recommendation_text(
            overall_fit, chest_cmp, shoulder_cmp, waist_cmp, hip_cmp, garment_spec
        )

        return FitAnalysis(
            body_measurements=body_spec,
            garment_measurements=garment_spec,
            chest_comparison=chest_cmp,
            shoulder_comparison=shoulder_cmp,
            waist_comparison=waist_cmp,
            hip_comparison=hip_cmp,
            overall_fit=overall_fit,
            fit_score=fit_score,
            recommendation=recommendation,
        )

    def rank_garments(
        self,
        body_spec: BodyMeasurements,
        garment_specs: List[GarmentMeasurements],
        min_score: float = 60.0,
    ) -> List[FitAnalysis]:
        """
        Analyze and rank multiple garments by fit score.

        Args:
            body_spec: Body measurements
            garment_specs: List of garment measurements
            min_score: Minimum fit score to include in results

        Returns:
            List of FitAnalysis sorted by fit_score (descending)
        """
        analyses = []

        for garment in garment_specs:
            analysis = self.analyze_fit(body_spec, garment)

            # Filter by minimum score
            if analysis.fit_score >= min_score:
                analyses.append(analysis)

        # Sort by fit score (highest first)
        analyses.sort(key=lambda a: a.fit_score, reverse=True)

        return analyses


def load_body_measurements_from_minio(
    session_id: str, clip_id: str, storage_client=None
) -> Optional[BodyMeasurements]:
    # Template data for development
    template_data = {
        "chest_circumference_cm": 95.0,
        "shoulder_width_cm": 45.0,
        "waist_circumference_cm": 82.0,
        "hip_circumference_cm": 98.0,
        "height_cm": 175.0,
        "session_id": session_id,
        "clip_id": clip_id,
    }

    return BodyMeasurements.from_json(template_data)


def load_garment_measurements_from_db(
    garment_ids: List[str], db_connection=None
) -> List[GarmentMeasurements]:
    # Template data for development
    template_garments = [
        {
            "garment_id": "G001",
            "size": "M",
            "chest_cm": 96.0,
            "shoulder_cm": 45.5,
            "waist_cm": 83.0,
            "hip_cm": 99.0,
            "category": "shirt",
            "brand": "Nike",
        },
        {
            "garment_id": "G002",
            "size": "L",
            "chest_cm": 100.0,
            "shoulder_cm": 47.0,
            "waist_cm": 87.0,
            "hip_cm": 103.0,
            "category": "shirt",
            "brand": "Adidas",
        },
        {
            "garment_id": "G003",
            "size": "S",
            "chest_cm": 92.0,
            "shoulder_cm": 43.0,
            "waist_cm": 79.0,
            "hip_cm": 95.0,
            "category": "shirt",
            "brand": "Puma",
        },
    ]

    return [GarmentMeasurements.from_json(g) for g in template_garments]


def get_recommendations_for_session(
    session_id: str,
    clip_id: str,
    candidate_garment_ids: List[str],
    storage_client=None,
    db_connection=None,
    min_score: float = 60.0,
) -> List[Dict]:
    """
    Complete recommendation pipeline: load measurements and rank garments.

    Args:
        session_id: User's capture session ID
        clip_id: Specific clip ID
        candidate_garment_ids: List of garment IDs to consider
        storage_client: MinIO/S3 client
        db_connection: Database connection
        min_score: Minimum fit score threshold

    Returns:
        List of recommendation dictionaries sorted by fit score
    """
    # Load body measurements from MinIO
    body_measurements = load_body_measurements_from_minio(
        session_id, clip_id, storage_client
    )

    if body_measurements is None:
        return []

    # Load garment measurements from database
    garment_measurements = load_garment_measurements_from_db(
        candidate_garment_ids, db_connection
    )

    if not garment_measurements:
        return []

    # Create recommender and rank garments
    recommender = FitRecommender()
    analyses = recommender.rank_garments(
        body_measurements,
        garment_measurements,
        min_score=min_score,
    )

    # Convert to dictionaries for JSON serialization
    return [analysis.to_dict() for analysis in analyses]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create body measurements from JSON
    body_json = {
        "chest_circumference_cm": 95.0,
        "shoulder_width_cm": 45.0,
        "waist_circumference_cm": 82.0,
        "hip_circumference_cm": 98.0,
        "height_cm": 175.0,
        "session_id": "session_123",
        "clip_id": "clip_456",
    }
    body = BodyMeasurements.from_json(body_json)

    # Example: Create garment measurements from JSON
    garment_json = {
        "garment_id": "G12345",
        "size": "M",
        "chest_cm": 96.0,
        "shoulder_cm": 45.5,
        "waist_cm": 83.0,
        "hip_cm": 99.0,
        "category": "shirt",
        "brand": "Nike",
    }
    garment = GarmentMeasurements.from_json(garment_json)

    # Example: Perform fit analysis
    recommender = FitRecommender()
    analysis = recommender.analyze_fit(body, garment)

    # Print results
    print(
        f"Overall Fit: {analysis.overall_fit.value} ({'FIT' if analysis.overall_fit.value == 1 else 'NO FIT'})"
    )
    print(f"Fit Score: {analysis.fit_score}/100")
    print(f"Recommendation: {analysis.recommendation}")
    print("\nMeasurement Details:")
    print(
        f"  Chest: {analysis.chest_comparison.difference:+.1f}cm (acceptable={analysis.chest_comparison.fit_status.value})"
    )
    print(
        f"  Shoulder: {analysis.shoulder_comparison.difference:+.1f}cm (acceptable={analysis.shoulder_comparison.fit_status.value})"
    )
    print(
        f"  Waist: {analysis.waist_comparison.difference:+.1f}cm (acceptable={analysis.waist_comparison.fit_status.value})"
    )
    print(
        f"  Hip: {analysis.hip_comparison.difference:+.1f}cm (acceptable={analysis.hip_comparison.fit_status.value})"
    )

    # Example: Rank multiple garments
    print("\n" + "=" * 70)
    print("RANKING MULTIPLE GARMENTS")
    print("=" * 70)

    garments = load_garment_measurements_from_db(["G001", "G002", "G003"])
    ranked = recommender.rank_garments(body, garments, min_score=60.0)

    for i, result in enumerate(ranked, 1):
        fit_label = "FIT" if result.overall_fit.value == 1 else "NO FIT"
        print(
            f"\n{i}. {result.garment_measurements.brand} - {result.garment_measurements.garment_id} (Size {result.garment_measurements.size})"
        )
        print(
            f"   Fit: {result.overall_fit.value} ({fit_label}) - Score: {result.fit_score}/100"
        )
        print(f"   {result.recommendation}")
