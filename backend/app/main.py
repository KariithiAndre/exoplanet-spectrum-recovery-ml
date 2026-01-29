"""
Exoplanet Spectrum Recovery API

A FastAPI-based backend for exoplanet atmospheric spectrum analysis and recovery.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import spectrum, models, analyses, health
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info("Starting Exoplanet Spectrum Recovery API...")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Load ML models on startup (if needed)
    # from app.services.ml_service import load_models
    # await load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Exoplanet Spectrum Recovery API",
    description="""
    A research-grade API for analyzing and recovering exoplanet atmospheric spectra 
    using deep learning and Bayesian inference.
    
    ## Features
    
    * **Spectrum Upload & Processing**: Support for FITS, CSV, and JSON formats
    * **ML-Powered Recovery**: Deep learning models for spectral denoising
    * **Atmospheric Retrieval**: Bayesian parameter estimation
    * **Interactive Analysis**: Real-time spectral analysis and visualization
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(spectrum.router, prefix="/api/spectrum", tags=["Spectrum"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(analyses.router, prefix="/api/analyses", tags=["Analyses"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Exoplanet Spectrum Recovery API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
    }
