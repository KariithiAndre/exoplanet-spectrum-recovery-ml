"""Pydantic models for ML model metadata."""

from pydantic import BaseModel, Field
from typing import List, Optional


class ModelInfo(BaseModel):
    """Information about an ML model."""
    
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: str = Field(..., description="Model description")
    version: str = Field(..., description="Model version")
    accuracy: float = Field(..., ge=0, le=100, description="Model accuracy percentage")
    status: str = Field(..., description="Model status (active, beta, deprecated)")
    architecture: str = Field(..., description="Model architecture type")
    input_shape: List[int] = Field(..., description="Expected input tensor shape")
    trained_on: str = Field(..., description="Training dataset description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "spectrum_denoiser_v2",
                "name": "Spectrum Denoiser v2",
                "description": "Transformer-based architecture for spectral denoising",
                "version": "2.0.0",
                "accuracy": 94.7,
                "status": "active",
                "architecture": "Transformer",
                "input_shape": [1, 2048],
                "trained_on": "100,000 synthetic + 500 real spectra",
            }
        }


class ModelListResponse(BaseModel):
    """Response model for listing available models."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of models")
