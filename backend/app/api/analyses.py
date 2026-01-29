"""API router for analysis history and results."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

router = APIRouter()


@router.get("/")
async def list_analyses(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    List recent analyses with pagination.
    """
    # Mock data (in production, fetch from database)
    analyses = [
        {
            "id": "analysis-001",
            "spectrum_filename": "wasp-39b_nirspec.fits",
            "model_used": "spectrum_denoiser_v2",
            "created_at": "2026-01-28T10:30:00Z",
            "status": "completed",
            "snr": 45.2,
            "features_detected": 5,
        },
        {
            "id": "analysis-002",
            "spectrum_filename": "hd209458b_wfc3.csv",
            "model_used": "spectrum_denoiser_v2",
            "created_at": "2026-01-27T15:45:00Z",
            "status": "completed",
            "snr": 32.8,
            "features_detected": 3,
        },
        {
            "id": "analysis-003",
            "spectrum_filename": "trappist-1e_miri.fits",
            "model_used": "retrieval_net",
            "created_at": "2026-01-26T09:15:00Z",
            "status": "processing",
            "snr": None,
            "features_detected": None,
        },
    ]
    
    return {
        "analyses": analyses[offset:offset + limit],
        "total": len(analyses),
        "limit": limit,
        "offset": offset,
    }


@router.get("/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Get detailed results for a specific analysis.
    """
    # Mock data
    if analysis_id == "analysis-001":
        return {
            "id": analysis_id,
            "spectrum_filename": "wasp-39b_nirspec.fits",
            "model_used": "spectrum_denoiser_v2",
            "created_at": "2026-01-28T10:30:00Z",
            "status": "completed",
            "results": {
                "snr": 45.2,
                "confidence": 0.92,
                "processing_time_ms": 847,
                "features": [
                    {"molecule": "H2O", "wavelength": 1.4, "significance": 4.5},
                    {"molecule": "CO2", "wavelength": 4.3, "significance": 3.2},
                    {"molecule": "CH4", "wavelength": 3.3, "significance": 2.8},
                ],
            },
        }
    
    raise HTTPException(status_code=404, detail="Analysis not found")


@router.get("/{analysis_id}/export")
async def export_analysis(
    analysis_id: str,
    format: str = Query(default="csv", regex="^(csv|json|fits)$"),
):
    """
    Export analysis results in the specified format.
    """
    # In production, generate and return the file
    raise HTTPException(status_code=501, detail="Export not yet implemented")
