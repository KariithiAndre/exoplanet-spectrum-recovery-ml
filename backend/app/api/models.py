"""API router for ML model management."""

from fastapi import APIRouter, HTTPException
from typing import List

from app.models.ml_models import ModelInfo, ModelListResponse

router = APIRouter()

# Mock model registry (in production, load from config/database)
AVAILABLE_MODELS = [
    ModelInfo(
        id="spectrum_denoiser_v1",
        name="Spectrum Denoiser v1",
        description="Convolutional neural network for spectral noise reduction",
        version="1.0.0",
        accuracy=92.3,
        status="active",
        architecture="CNN",
        input_shape=[1, 1024],
        trained_on="50,000 synthetic spectra",
    ),
    ModelInfo(
        id="spectrum_denoiser_v2",
        name="Spectrum Denoiser v2",
        description="Transformer-based architecture with attention mechanisms",
        version="2.0.0",
        accuracy=94.7,
        status="active",
        architecture="Transformer",
        input_shape=[1, 2048],
        trained_on="100,000 synthetic + 500 real spectra",
    ),
    ModelInfo(
        id="retrieval_net",
        name="Retrieval Network",
        description="End-to-end atmospheric parameter estimation",
        version="1.0.0",
        accuracy=89.1,
        status="beta",
        architecture="MLP + Attention",
        input_shape=[1, 1024],
        trained_on="200,000 petitRADTRANS models",
    ),
    ModelInfo(
        id="feature_detector",
        name="Feature Detector",
        description="Molecular absorption feature identification",
        version="1.2.0",
        accuracy=96.2,
        status="active",
        architecture="ResNet",
        input_shape=[1, 1024],
        trained_on="75,000 labeled spectra",
    ),
]


@router.get("/", response_model=ModelListResponse)
async def list_models():
    """
    List all available ML models.
    """
    return ModelListResponse(
        models=AVAILABLE_MODELS,
        total=len(AVAILABLE_MODELS),
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get details about a specific model.
    """
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    
    raise HTTPException(status_code=404, detail="Model not found")


@router.post("/{model_id}/load")
async def load_model(model_id: str):
    """
    Load a model into memory for inference.
    """
    # In production, this would actually load the model weights
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return {"status": "loaded", "model_id": model_id}
    
    raise HTTPException(status_code=404, detail="Model not found")
