"""Pydantic models for spectrum data."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SpectrumData(BaseModel):
    """Representation of spectral data."""
    
    wavelength: List[float] = Field(..., description="Wavelength values in microns")
    flux: List[float] = Field(..., description="Flux/transit depth values in ppm")
    error: Optional[List[float]] = Field(None, description="Uncertainty values")
    recovered: Optional[List[float]] = Field(None, description="Recovered/denoised flux")
    model: Optional[List[float]] = Field(None, description="Best-fit model flux")
    
    class Config:
        json_schema_extra = {
            "example": {
                "wavelength": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "flux": [100.2, 102.5, 98.7, 105.3, 101.8, 99.4],
                "error": [5.1, 4.8, 5.3, 4.9, 5.0, 5.2],
            }
        }


class AnalysisResult(BaseModel):
    """Results from spectrum analysis."""
    
    snr: float = Field(..., description="Signal-to-noise ratio")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    processing_time: int = Field(..., description="Processing time in milliseconds")
    features: List[dict] = Field(default_factory=list, description="Detected spectral features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "snr": 45.2,
                "confidence": 0.92,
                "processing_time": 847,
                "features": [
                    {"molecule": "H2O", "wavelength": 1.4, "significance": 4.5}
                ],
            }
        }


class SpectrumResponse(BaseModel):
    """Response model for spectrum analysis endpoint."""
    
    id: str = Field(..., description="Unique identifier for this analysis")
    filename: str = Field(..., description="Original filename")
    spectrum: SpectrumData = Field(..., description="Processed spectrum data")
    analysis: AnalysisResult = Field(..., description="Analysis results")
    created_at: datetime = Field(..., description="Timestamp of analysis")


class SpectrumUploadRequest(BaseModel):
    """Request model for spectrum upload."""
    
    wavelength_unit: str = Field(default="micron", description="Unit of wavelength values")
    flux_unit: str = Field(default="ppm", description="Unit of flux values")
    instrument: Optional[str] = Field(None, description="Instrument used for observation")
    target_name: Optional[str] = Field(None, description="Name of the target exoplanet")
