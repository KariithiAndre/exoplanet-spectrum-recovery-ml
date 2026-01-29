"""API router for health check endpoints."""

from fastapi import APIRouter
import torch
import platform
from datetime import datetime

router = APIRouter()


@router.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns system status including GPU availability and memory usage.
    """
    gpu_available = torch.cuda.is_available()
    gpu_info = None
    
    if gpu_available:
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        },
        "gpu": {
            "available": gpu_available,
            "info": gpu_info,
        },
    }
