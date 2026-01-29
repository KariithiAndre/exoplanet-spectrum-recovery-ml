"""API router for spectrum processing endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import uuid
import numpy as np
from datetime import datetime

from app.models.spectrum import SpectrumData, SpectrumResponse, AnalysisResult
from app.services.spectrum_service import SpectrumService

router = APIRouter()
spectrum_service = SpectrumService()


@router.post("/analyze", response_model=SpectrumResponse)
async def analyze_spectrum(file: UploadFile = File(...)):
    """
    Upload and analyze a spectrum file.
    
    Accepts FITS, CSV, or JSON format files containing spectral data.
    Returns the processed spectrum data along with initial analysis results.
    """
    # Validate file type
    allowed_extensions = {".fits", ".fit", ".csv", ".json"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Process spectrum
        spectrum_data = await spectrum_service.process_file(content, file_ext)
        
        # Run initial analysis
        analysis = await spectrum_service.analyze(spectrum_data)
        
        return SpectrumResponse(
            id=str(uuid.uuid4()),
            filename=file.filename,
            spectrum=spectrum_data,
            analysis=analysis,
            created_at=datetime.utcnow(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{spectrum_id}")
async def get_spectrum(spectrum_id: str):
    """
    Retrieve a previously analyzed spectrum by ID.
    """
    # In a real implementation, this would fetch from database
    raise HTTPException(status_code=404, detail="Spectrum not found")


@router.post("/recover")
async def recover_spectrum(
    spectrum_id: str,
    model_id: str = "spectrum_denoiser_v2",
    wavelength_min: Optional[float] = None,
    wavelength_max: Optional[float] = None,
):
    """
    Run spectral recovery using a trained ML model.
    
    Applies the specified denoising/recovery model to extract
    atmospheric features from the noisy input spectrum.
    """
    try:
        result = await spectrum_service.recover(
            spectrum_id=spectrum_id,
            model_id=model_id,
            wavelength_range=(wavelength_min, wavelength_max),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
