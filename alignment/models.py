"""
Feature-ontology alignment data models.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FeatureMappingRule(BaseModel):
    """A single feature-to-subdomain mapping rule."""

    detector: str = Field(..., description="Detector name, e.g. FFTDetector")
    feature: str = Field(..., description="Feature name, e.g. jpeg_blockiness")
    subdomain_id: str = Field(..., description="SubDomain node id")
    subdomain_label: str = Field(..., description="SubDomain display name")
    evidence_enabled: bool = Field(
        default=True,
        description="Whether this rule should activate graph evidence during alignment",
    )
    sigmoid_k: float = Field(default=10.0, description="Sigmoid slope")
    sigmoid_x0: float = Field(default=0.5, description="Sigmoid midpoint")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Feature weight after score mapping",
    )
    activation_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum final confidence required to activate the subdomain",
    )
    context_detector: Optional[str] = Field(
        default=None,
        description="Optional detector name used for context gating",
    )
    context_feature: Optional[str] = Field(
        default=None,
        description="Optional feature name used for context gating",
    )
    context_min_value: float = Field(
        default=0.0,
        description="Minimum context feature value required before activation",
    )


class MappingConfig(BaseModel):
    """Mapping config file schema."""

    version: str = Field(default="1.0", description="Config version")
    rules: List[FeatureMappingRule] = Field(default_factory=list)


class ActivatedSubDomain(BaseModel):
    """Activated subdomain payload produced by the aligner."""

    model_config = ConfigDict(ser_json_inf_nan="constants")

    subdomain_id: str = Field(..., description="SubDomain node id")
    subdomain_label: str = Field(..., description="SubDomain display name")
    score: float = Field(..., ge=0.0, le=1.0, description="Mapped sigmoid score")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final confidence computed as score * weight",
    )
    source_detector: str = Field(..., description="Source detector name")
    source_feature: str = Field(..., description="Source feature name")
    raw_value: float = Field(..., description="Raw feature value")
