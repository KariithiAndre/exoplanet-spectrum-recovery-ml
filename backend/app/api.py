"""
FastAPI Backend for Exoplanet Spectrum Analysis

Provides REST API endpoints for:
1. Spectrum upload and management
2. Synthetic spectrum simulation
3. Preprocessing pipeline execution
4. ML model inference (CNN/Transformer)
5. Explainability and interpretation
6. Results export (JSON/PDF)

Author: Exoplanet Spectrum Recovery Project
"""

import io
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="Exoplanet Spectrum Analysis API",
    description="API for analyzing exoplanet transmission spectra using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")
MODELS_DIR = Path("models/checkpoints")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Pydantic Models
# =============================================================================

class InstrumentType(str, Enum):
    JWST_NIRSPEC = "jwst_nirspec"
    JWST_MIRI = "jwst_miri"
    JWST_NIRCAM = "jwst_nircam"
    HUBBLE_WFC3 = "hubble_wfc3"
    HUBBLE_STIS = "hubble_stis"
    GENERIC = "generic"


class ModelType(str, Enum):
    CNN = "cnn"
    TRANSFORMER = "transformer"


class PreprocessingMethod(str, Enum):
    SAVGOL = "savgol"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    LOWESS = "lowess"


class BaselineMethod(str, Enum):
    POLYNOMIAL = "polynomial"
    SPLINE = "spline"
    ALS = "als"


class ExplainMethod(str, Enum):
    GRADIENT = "gradient"
    SMOOTH_GRAD = "smooth_grad"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    OCCLUSION = "occlusion"


# Request Models
class SpectrumData(BaseModel):
    """Raw spectrum data input."""
    wavelengths: List[float] = Field(..., description="Wavelength values in microns")
    flux: List[float] = Field(..., description="Flux or transit depth values")
    errors: Optional[List[float]] = Field(None, description="Measurement errors")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('flux')
    def validate_flux_length(cls, v, values):
        if 'wavelengths' in values and len(v) != len(values['wavelengths']):
            raise ValueError('flux and wavelengths must have same length')
        return v


class SyntheticSpectrumRequest(BaseModel):
    """Request for synthetic spectrum generation."""
    planet_type: str = Field("hot_jupiter", description="Type of planet")
    star_type: str = Field("G2V", description="Stellar type")
    molecules: List[str] = Field(default_factory=lambda: ["H2O", "CO2"], description="Molecules to include")
    abundances: Optional[Dict[str, float]] = Field(None, description="Molecular abundances (log10)")
    temperature: float = Field(1500, description="Atmospheric temperature (K)")
    wavelength_range: tuple = Field((0.6, 5.0), description="Wavelength range in microns")
    resolution: int = Field(512, description="Number of wavelength points")
    add_noise: bool = Field(True, description="Add realistic noise")
    noise_level: float = Field(0.01, description="Noise standard deviation")
    instrument: InstrumentType = Field(InstrumentType.JWST_NIRSPEC)


class PreprocessingRequest(BaseModel):
    """Preprocessing configuration."""
    spectrum_id: Optional[str] = Field(None, description="ID of uploaded spectrum")
    spectrum_data: Optional[SpectrumData] = Field(None, description="Direct spectrum data")
    
    # Smoothing
    smoothing_method: PreprocessingMethod = Field(PreprocessingMethod.SAVGOL)
    smoothing_window: int = Field(11, ge=3, le=51)
    smoothing_order: int = Field(3, ge=1, le=5)
    
    # Baseline correction
    baseline_method: BaselineMethod = Field(BaselineMethod.POLYNOMIAL)
    baseline_order: int = Field(3, ge=1, le=10)
    
    # Outlier removal
    remove_outliers: bool = Field(True)
    outlier_sigma: float = Field(3.0, ge=1.0, le=10.0)
    
    # Normalization
    normalize: bool = Field(True)
    normalization_method: str = Field("minmax")


class InferenceRequest(BaseModel):
    """ML inference request."""
    spectrum_id: Optional[str] = Field(None, description="ID of uploaded/processed spectrum")
    spectrum_data: Optional[SpectrumData] = Field(None, description="Direct spectrum data")
    model_type: ModelType = Field(ModelType.TRANSFORMER)
    with_uncertainty: bool = Field(True, description="Include uncertainty estimates")
    preprocess: bool = Field(True, description="Apply preprocessing before inference")


class ExplainRequest(BaseModel):
    """Explainability request."""
    spectrum_id: Optional[str] = Field(None)
    spectrum_data: Optional[SpectrumData] = Field(None)
    model_type: ModelType = Field(ModelType.TRANSFORMER)
    method: ExplainMethod = Field(ExplainMethod.INTEGRATED_GRADIENTS)
    target_molecule: Optional[str] = Field(None, description="Specific molecule to explain")
    detect_regions: bool = Field(True, description="Detect influential regions")


class ExportRequest(BaseModel):
    """Export request."""
    spectrum_id: str = Field(..., description="Spectrum ID to export")
    format: str = Field("json", description="Export format: json or pdf")
    include_plots: bool = Field(True)
    include_raw_data: bool = Field(True)
    include_predictions: bool = Field(True)
    include_explanations: bool = Field(True)


class BatchInferenceRequest(BaseModel):
    """Batch inference request."""
    spectrum_ids: List[str] = Field(..., description="List of spectrum IDs")
    model_type: ModelType = Field(ModelType.TRANSFORMER)
    with_uncertainty: bool = Field(True)


# Response Models
class SpectrumResponse(BaseModel):
    """Spectrum upload/retrieval response."""
    id: str
    filename: Optional[str]
    wavelength_range: tuple
    num_points: int
    instrument: Optional[str]
    upload_time: str
    metadata: Dict[str, Any]


class PreprocessingResponse(BaseModel):
    """Preprocessing result."""
    spectrum_id: str
    original_points: int
    processed_points: int
    outliers_removed: int
    processing_steps: List[str]
    wavelengths: List[float]
    flux: List[float]
    processing_time_ms: float


class PredictionResponse(BaseModel):
    """ML prediction response."""
    spectrum_id: str
    model_type: str
    
    # Molecule detection
    detected_molecules: List[str]
    molecule_probabilities: Dict[str, float]
    molecule_uncertainties: Optional[Dict[str, float]]
    
    # Planet classification
    planet_class: str
    planet_probabilities: Dict[str, float]
    planet_uncertainty: Optional[float]
    
    # Habitability
    habitability_score: float
    habitability_uncertainty: Optional[float]
    
    # Metadata
    inference_time_ms: float
    model_version: str


class ExplainResponse(BaseModel):
    """Explainability response."""
    spectrum_id: str
    method: str
    
    # Saliency data
    saliency: List[float]
    wavelengths: List[float]
    
    # Influential regions
    regions: List[Dict[str, Any]]
    
    # Per-molecule attributions
    molecule_attributions: Optional[Dict[str, List[float]]]
    
    processing_time_ms: float


class ExportResponse(BaseModel):
    """Export response."""
    spectrum_id: str
    format: str
    download_url: str
    file_size_bytes: int
    expires_at: str


# =============================================================================
# In-Memory Storage (Replace with database in production)
# =============================================================================

class SpectrumStore:
    """Simple in-memory spectrum storage."""
    
    def __init__(self):
        self.spectra: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, Dict[str, Any]] = {}
        self.explanations: Dict[str, Dict[str, Any]] = {}
    
    def save_spectrum(
        self,
        wavelengths: np.ndarray,
        flux: np.ndarray,
        errors: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Save spectrum and return ID."""
        spectrum_id = str(uuid.uuid4())[:8]
        
        self.spectra[spectrum_id] = {
            'id': spectrum_id,
            'wavelengths': wavelengths,
            'flux': flux,
            'errors': errors,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
        }
        
        return spectrum_id
    
    def get_spectrum(self, spectrum_id: str) -> Optional[Dict]:
        return self.spectra.get(spectrum_id)
    
    def save_prediction(self, spectrum_id: str, prediction: Dict):
        self.predictions[spectrum_id] = prediction
    
    def get_prediction(self, spectrum_id: str) -> Optional[Dict]:
        return self.predictions.get(spectrum_id)
    
    def save_explanation(self, spectrum_id: str, explanation: Dict):
        self.explanations[spectrum_id] = explanation
    
    def get_explanation(self, spectrum_id: str) -> Optional[Dict]:
        return self.explanations.get(spectrum_id)
    
    def list_spectra(self, limit: int = 100) -> List[Dict]:
        return list(self.spectra.values())[:limit]


# Global store
store = SpectrumStore()


# =============================================================================
# Utility Functions
# =============================================================================

def generate_synthetic_spectrum(config: SyntheticSpectrumRequest) -> Dict[str, np.ndarray]:
    """Generate synthetic exoplanet spectrum."""
    
    # Generate wavelength grid
    wavelengths = np.linspace(
        config.wavelength_range[0],
        config.wavelength_range[1],
        config.resolution
    )
    
    # Base continuum
    flux = np.ones_like(wavelengths)
    
    # Molecular absorption bands (simplified model)
    MOLECULE_FEATURES = {
        "H2O": [(1.4, 0.15, 0.05), (1.9, 0.12, 0.04), (2.7, 0.18, 0.06)],
        "CO2": [(2.0, 0.10, 0.03), (2.7, 0.08, 0.02), (4.3, 0.25, 0.08)],
        "CO": [(2.35, 0.08, 0.02), (4.6, 0.15, 0.05)],
        "CH4": [(1.7, 0.10, 0.04), (2.3, 0.12, 0.03), (3.3, 0.20, 0.06)],
        "NH3": [(1.5, 0.08, 0.03), (2.0, 0.06, 0.02), (10.5, 0.15, 0.08)],
        "O3": [(9.6, 0.12, 0.10)],
        "Na": [(0.589, 0.05, 0.001)],
        "K": [(0.767, 0.04, 0.001)],
    }
    
    # Add molecular features
    for molecule in config.molecules:
        if molecule in MOLECULE_FEATURES:
            # Get abundance scaling
            abundance = 1.0
            if config.abundances and molecule in config.abundances:
                abundance = 10 ** (config.abundances[molecule] + 3)  # Scale
            
            for center, depth, width in MOLECULE_FEATURES[molecule]:
                if config.wavelength_range[0] <= center <= config.wavelength_range[1]:
                    # Temperature-dependent depth scaling
                    temp_scale = min(config.temperature / 1500, 2.0)
                    scaled_depth = depth * abundance * temp_scale
                    
                    # Gaussian absorption profile
                    flux -= scaled_depth * np.exp(
                        -((wavelengths - center) ** 2) / (2 * width ** 2)
                    )
    
    # Add noise if requested
    if config.add_noise:
        noise = np.random.normal(0, config.noise_level, len(flux))
        flux += noise
        errors = np.full_like(flux, config.noise_level)
    else:
        errors = np.zeros_like(flux)
    
    return {
        'wavelengths': wavelengths,
        'flux': flux,
        'errors': errors,
    }


def apply_preprocessing(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    config: PreprocessingRequest,
) -> Dict[str, Any]:
    """Apply preprocessing pipeline."""
    from scipy import signal
    from scipy.ndimage import median_filter
    
    steps = []
    original_len = len(flux)
    processed_flux = flux.copy()
    
    # 1. Outlier removal
    outliers_removed = 0
    if config.remove_outliers:
        median = np.median(processed_flux)
        std = np.std(processed_flux)
        mask = np.abs(processed_flux - median) < config.outlier_sigma * std
        
        outliers_removed = np.sum(~mask)
        
        # Interpolate outliers
        if outliers_removed > 0:
            good_indices = np.where(mask)[0]
            bad_indices = np.where(~mask)[0]
            processed_flux[bad_indices] = np.interp(
                bad_indices, good_indices, processed_flux[good_indices]
            )
            steps.append(f"Removed {outliers_removed} outliers (σ={config.outlier_sigma})")
    
    # 2. Smoothing
    if config.smoothing_method == PreprocessingMethod.SAVGOL:
        processed_flux = signal.savgol_filter(
            processed_flux,
            window_length=config.smoothing_window,
            polyorder=config.smoothing_order,
        )
        steps.append(f"Savitzky-Golay smoothing (window={config.smoothing_window})")
    
    elif config.smoothing_method == PreprocessingMethod.GAUSSIAN:
        from scipy.ndimage import gaussian_filter1d
        sigma = config.smoothing_window / 4
        processed_flux = gaussian_filter1d(processed_flux, sigma)
        steps.append(f"Gaussian smoothing (σ={sigma:.1f})")
    
    elif config.smoothing_method == PreprocessingMethod.MEDIAN:
        processed_flux = median_filter(processed_flux, size=config.smoothing_window)
        steps.append(f"Median filter (window={config.smoothing_window})")
    
    # 3. Baseline correction
    if config.baseline_method == BaselineMethod.POLYNOMIAL:
        coeffs = np.polyfit(wavelengths, processed_flux, config.baseline_order)
        baseline = np.polyval(coeffs, wavelengths)
        processed_flux = processed_flux - baseline + np.median(baseline)
        steps.append(f"Polynomial baseline (order={config.baseline_order})")
    
    # 4. Normalization
    if config.normalize:
        if config.normalization_method == "minmax":
            min_val, max_val = processed_flux.min(), processed_flux.max()
            processed_flux = (processed_flux - min_val) / (max_val - min_val + 1e-8)
            steps.append("Min-max normalization")
        elif config.normalization_method == "zscore":
            processed_flux = (processed_flux - np.mean(processed_flux)) / (np.std(processed_flux) + 1e-8)
            steps.append("Z-score normalization")
    
    return {
        'wavelengths': wavelengths,
        'flux': processed_flux,
        'steps': steps,
        'outliers_removed': outliers_removed,
    }


def run_inference(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    model_type: ModelType,
    with_uncertainty: bool = True,
) -> Dict[str, Any]:
    """Run ML model inference."""
    
    # In production, load actual models
    # For demo, return mock predictions based on spectral features
    
    # Analyze spectrum for features
    flux_normalized = (flux - flux.min()) / (flux.max() - flux.min() + 1e-8)
    
    # Mock molecule detection based on absorption features
    MOLECULE_CENTERS = {
        "H2O": [1.4, 1.9, 2.7],
        "CO2": [2.0, 2.7, 4.3],
        "CO": [2.35, 4.6],
        "CH4": [1.7, 2.3, 3.3],
        "NH3": [1.5, 2.0],
        "O2": [0.76, 1.27],
        "O3": [9.6],
        "Na": [0.589],
        "K": [0.767],
        "TiO": [0.8],
        "VO": [0.75],
        "FeH": [1.0],
        "H2S": [2.5],
        "HCN": [3.0],
    }
    
    molecule_probs = {}
    molecule_uncertainties = {}
    
    for mol, centers in MOLECULE_CENTERS.items():
        prob = 0.1  # Base probability
        
        for center in centers:
            if wavelengths.min() <= center <= wavelengths.max():
                idx = np.argmin(np.abs(wavelengths - center))
                
                # Check for absorption feature
                window = slice(max(0, idx - 5), min(len(flux), idx + 5))
                local_min = flux_normalized[window].min()
                surrounding = np.median(flux_normalized)
                
                if local_min < surrounding - 0.05:
                    prob += 0.3 * (surrounding - local_min)
        
        prob = min(prob, 0.99)
        molecule_probs[mol] = float(prob)
        
        if with_uncertainty:
            molecule_uncertainties[mol] = float(np.random.uniform(0.05, 0.15))
    
    # Detected molecules
    detected = [mol for mol, prob in molecule_probs.items() if prob > 0.5]
    
    # Planet classification (mock)
    planet_classes = [
        "HOT_JUPITER", "WARM_JUPITER", "COLD_JUPITER", "HOT_NEPTUNE",
        "WARM_NEPTUNE", "SUPER_EARTH", "EARTH_LIKE", "WATER_WORLD",
        "LAVA_WORLD", "UNKNOWN"
    ]
    
    # Heuristic classification
    if "H2O" in detected and "O3" in detected:
        main_class = "EARTH_LIKE"
    elif "H2O" in detected and len(detected) > 3:
        main_class = "WATER_WORLD"
    elif "TiO" in detected or "VO" in detected:
        main_class = "HOT_JUPITER"
    elif "CH4" in detected:
        main_class = "WARM_JUPITER"
    else:
        main_class = "UNKNOWN"
    
    planet_probs = {c: 0.05 for c in planet_classes}
    planet_probs[main_class] = 0.6 + np.random.uniform(0, 0.3)
    
    # Normalize
    total = sum(planet_probs.values())
    planet_probs = {k: v / total for k, v in planet_probs.items()}
    
    # Habitability score
    hab_indicators = ["H2O", "O2", "O3", "CO2"]
    hab_score = sum(molecule_probs.get(m, 0) for m in hab_indicators) / len(hab_indicators)
    hab_score = min(hab_score, 1.0)
    
    return {
        'detected_molecules': detected,
        'molecule_probabilities': molecule_probs,
        'molecule_uncertainties': molecule_uncertainties if with_uncertainty else None,
        'planet_class': main_class,
        'planet_probabilities': planet_probs,
        'planet_uncertainty': float(np.random.uniform(0.05, 0.15)) if with_uncertainty else None,
        'habitability_score': float(hab_score),
        'habitability_uncertainty': float(np.random.uniform(0.02, 0.08)) if with_uncertainty else None,
        'model_version': f"{model_type.value}_v1.0",
    }


def compute_saliency(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    method: ExplainMethod,
    target_molecule: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute saliency/attribution for spectrum."""
    
    # In production, use actual gradient computation
    # For demo, generate feature-based saliency
    
    flux_normalized = (flux - flux.min()) / (flux.max() - flux.min() + 1e-8)
    
    # Base saliency: gradient of flux
    gradient = np.gradient(flux_normalized)
    
    # Find absorption features
    saliency = np.zeros_like(flux)
    
    # Detect local minima (absorption features)
    for i in range(2, len(flux) - 2):
        if flux_normalized[i] < flux_normalized[i-1] and flux_normalized[i] < flux_normalized[i+1]:
            # Absorption feature
            depth = np.median(flux_normalized) - flux_normalized[i]
            
            # Add saliency around feature
            width = 5
            for j in range(max(0, i - width), min(len(flux), i + width)):
                dist = abs(j - i)
                saliency[j] += depth * np.exp(-dist / 3)
    
    # Add noise for realism
    if method == ExplainMethod.SMOOTH_GRAD:
        saliency += np.random.normal(0, 0.02, len(saliency))
    
    # Normalize
    saliency = saliency / (np.abs(saliency).max() + 1e-8)
    
    # Detect influential regions
    regions = []
    threshold = 0.1
    
    in_region = False
    start_idx = 0
    
    for i, sal in enumerate(saliency):
        if sal > threshold and not in_region:
            start_idx = i
            in_region = True
        elif sal <= threshold and in_region:
            regions.append({
                'start_idx': start_idx,
                'end_idx': i - 1,
                'start_wavelength': float(wavelengths[start_idx]),
                'end_wavelength': float(wavelengths[i - 1]),
                'center_wavelength': float(wavelengths[(start_idx + i - 1) // 2]),
                'max_importance': float(saliency[start_idx:i].max()),
                'possible_molecules': [],
            })
            in_region = False
    
    # Match to molecular bands
    MOLECULAR_BANDS = {
        "H2O": [(1.35, 1.45), (1.85, 1.95), (2.65, 2.75)],
        "CO2": [(2.65, 2.75), (4.2, 4.4)],
        "CH4": [(1.65, 1.75), (2.25, 2.35), (3.25, 3.35)],
        "CO": [(2.3, 2.4), (4.55, 4.65)],
    }
    
    for region in regions:
        center = region['center_wavelength']
        for mol, bands in MOLECULAR_BANDS.items():
            for band_start, band_end in bands:
                if band_start <= center <= band_end:
                    region['possible_molecules'].append(mol)
    
    return {
        'saliency': saliency,
        'regions': regions,
    }


def export_to_json(spectrum_id: str) -> Dict[str, Any]:
    """Export all results to JSON."""
    spectrum = store.get_spectrum(spectrum_id)
    prediction = store.get_prediction(spectrum_id)
    explanation = store.get_explanation(spectrum_id)
    
    if not spectrum:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    
    export_data = {
        'spectrum_id': spectrum_id,
        'export_time': datetime.now().isoformat(),
        'spectrum': {
            'wavelengths': spectrum['wavelengths'].tolist(),
            'flux': spectrum['flux'].tolist(),
            'errors': spectrum['errors'].tolist() if spectrum.get('errors') is not None else None,
            'metadata': spectrum['metadata'],
        },
    }
    
    if prediction:
        export_data['prediction'] = prediction
    
    if explanation:
        export_data['explanation'] = {
            'saliency': explanation.get('saliency', []),
            'regions': explanation.get('regions', []),
        }
    
    return export_data


def export_to_pdf(spectrum_id: str) -> bytes:
    """Export results to PDF with plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    spectrum = store.get_spectrum(spectrum_id)
    prediction = store.get_prediction(spectrum_id)
    explanation = store.get_explanation(spectrum_id)
    
    if not spectrum:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    
    # Create PDF in memory
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Spectrum
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(spectrum['wavelengths'], spectrum['flux'], 'b-', linewidth=1)
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        ax.set_title(f'Exoplanet Spectrum - {spectrum_id}')
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2: Predictions
        if prediction:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Molecule probabilities
            ax1 = axes[0]
            mols = list(prediction['molecule_probabilities'].keys())
            probs = [prediction['molecule_probabilities'][m] for m in mols]
            colors = ['green' if p > 0.5 else 'gray' for p in probs]
            ax1.barh(mols, probs, color=colors)
            ax1.axvline(x=0.5, color='red', linestyle='--')
            ax1.set_xlabel('Detection Probability')
            ax1.set_title('Molecule Detection')
            
            # Planet class
            ax2 = axes[1]
            classes = list(prediction['planet_probabilities'].keys())
            probs = [prediction['planet_probabilities'][c] for c in classes]
            ax2.barh(classes, probs)
            ax2.set_xlabel('Probability')
            ax2.set_title(f"Planet Class: {prediction['planet_class']}")
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # Page 3: Explainability
        if explanation and 'saliency' in explanation:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            saliency = np.array(explanation['saliency'])
            
            ax1 = axes[0]
            ax1.plot(spectrum['wavelengths'], spectrum['flux'], 'b-')
            ax1.set_ylabel('Flux')
            ax1.set_title('Spectrum with Influential Regions')
            
            # Highlight regions
            for region in explanation.get('regions', []):
                ax1.axvspan(
                    region['start_wavelength'],
                    region['end_wavelength'],
                    alpha=0.3, color='red'
                )
            
            ax2 = axes[1]
            ax2.fill_between(spectrum['wavelengths'], 0, saliency, alpha=0.7)
            ax2.set_xlabel('Wavelength (μm)')
            ax2.set_ylabel('Importance')
            ax2.set_title('Saliency Map')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# API Endpoints
# =============================================================================

# Health check
@app.get("/", tags=["Health"])
async def root():
    """API health check."""
    return {
        "status": "healthy",
        "service": "Exoplanet Spectrum Analysis API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "spectra_count": len(store.spectra),
        "models_available": ["cnn", "transformer"],
    }


# =============================================================================
# Spectrum Upload Endpoints
# =============================================================================

@app.post("/api/v1/spectra/upload", response_model=SpectrumResponse, tags=["Spectra"])
async def upload_spectrum(
    file: UploadFile = File(...),
    instrument: InstrumentType = Query(InstrumentType.GENERIC),
):
    """
    Upload a spectrum file (CSV, FITS, JSON).
    
    Supports:
    - CSV with wavelength and flux columns
    - FITS files with spectral data
    - JSON with wavelengths and flux arrays
    """
    try:
        content = await file.read()
        filename = file.filename or "unknown"
        
        # Parse based on file type
        if filename.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # Find wavelength and flux columns
            wl_col = next((c for c in df.columns if 'wave' in c.lower()), df.columns[0])
            flux_col = next((c for c in df.columns if 'flux' in c.lower() or 'depth' in c.lower()), df.columns[1])
            
            wavelengths = df[wl_col].values
            flux = df[flux_col].values
            errors = df['error'].values if 'error' in df.columns else None
            
        elif filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            wavelengths = np.array(data['wavelengths'])
            flux = np.array(data['flux'])
            errors = np.array(data.get('errors')) if 'errors' in data else None
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate
        if len(wavelengths) != len(flux):
            raise HTTPException(status_code=400, detail="Wavelength and flux arrays must have same length")
        
        if len(wavelengths) < 10:
            raise HTTPException(status_code=400, detail="Spectrum too short (minimum 10 points)")
        
        # Save
        spectrum_id = store.save_spectrum(
            wavelengths=wavelengths,
            flux=flux,
            errors=errors,
            metadata={
                'filename': filename,
                'instrument': instrument.value,
            }
        )
        
        return SpectrumResponse(
            id=spectrum_id,
            filename=filename,
            wavelength_range=(float(wavelengths.min()), float(wavelengths.max())),
            num_points=len(wavelengths),
            instrument=instrument.value,
            upload_time=datetime.now().isoformat(),
            metadata={'instrument': instrument.value},
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/spectra/data", response_model=SpectrumResponse, tags=["Spectra"])
async def submit_spectrum_data(data: SpectrumData):
    """Submit spectrum data directly as JSON."""
    
    wavelengths = np.array(data.wavelengths)
    flux = np.array(data.flux)
    errors = np.array(data.errors) if data.errors else None
    
    spectrum_id = store.save_spectrum(
        wavelengths=wavelengths,
        flux=flux,
        errors=errors,
        metadata=data.metadata,
    )
    
    return SpectrumResponse(
        id=spectrum_id,
        filename=None,
        wavelength_range=(float(wavelengths.min()), float(wavelengths.max())),
        num_points=len(wavelengths),
        instrument=data.metadata.get('instrument'),
        upload_time=datetime.now().isoformat(),
        metadata=data.metadata,
    )


@app.get("/api/v1/spectra/{spectrum_id}", tags=["Spectra"])
async def get_spectrum(spectrum_id: str):
    """Retrieve spectrum data by ID."""
    spectrum = store.get_spectrum(spectrum_id)
    
    if not spectrum:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    
    return {
        'id': spectrum['id'],
        'wavelengths': spectrum['wavelengths'].tolist(),
        'flux': spectrum['flux'].tolist(),
        'errors': spectrum['errors'].tolist() if spectrum.get('errors') is not None else None,
        'metadata': spectrum['metadata'],
        'created_at': spectrum['created_at'],
    }


@app.get("/api/v1/spectra", tags=["Spectra"])
async def list_spectra(limit: int = Query(100, le=1000)):
    """List all uploaded spectra."""
    spectra = store.list_spectra(limit)
    
    return [
        {
            'id': s['id'],
            'wavelength_range': (float(s['wavelengths'].min()), float(s['wavelengths'].max())),
            'num_points': len(s['wavelengths']),
            'created_at': s['created_at'],
            'metadata': s['metadata'],
        }
        for s in spectra
    ]


@app.delete("/api/v1/spectra/{spectrum_id}", tags=["Spectra"])
async def delete_spectrum(spectrum_id: str):
    """Delete a spectrum."""
    if spectrum_id not in store.spectra:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    
    del store.spectra[spectrum_id]
    
    # Also delete associated data
    store.predictions.pop(spectrum_id, None)
    store.explanations.pop(spectrum_id, None)
    
    return {"status": "deleted", "id": spectrum_id}


# =============================================================================
# Synthetic Spectrum Generation
# =============================================================================

@app.post("/api/v1/synthetic/generate", response_model=SpectrumResponse, tags=["Synthetic"])
async def generate_spectrum(config: SyntheticSpectrumRequest):
    """
    Generate a synthetic exoplanet transmission spectrum.
    
    Simulates realistic spectral features based on:
    - Planet type and temperature
    - Atmospheric composition (molecular abundances)
    - Instrument characteristics
    """
    result = generate_synthetic_spectrum(config)
    
    spectrum_id = store.save_spectrum(
        wavelengths=result['wavelengths'],
        flux=result['flux'],
        errors=result['errors'],
        metadata={
            'synthetic': True,
            'planet_type': config.planet_type,
            'molecules': config.molecules,
            'temperature': config.temperature,
            'instrument': config.instrument.value,
        }
    )
    
    return SpectrumResponse(
        id=spectrum_id,
        filename=None,
        wavelength_range=config.wavelength_range,
        num_points=config.resolution,
        instrument=config.instrument.value,
        upload_time=datetime.now().isoformat(),
        metadata={
            'synthetic': True,
            'planet_type': config.planet_type,
            'molecules': config.molecules,
        },
    )


@app.get("/api/v1/synthetic/templates", tags=["Synthetic"])
async def list_templates():
    """List available planet templates for synthetic generation."""
    return {
        "templates": [
            {
                "name": "hot_jupiter",
                "description": "Hot Jupiter with high temperature atmosphere",
                "default_molecules": ["H2O", "CO", "TiO", "VO", "Na", "K"],
                "temperature_range": [1500, 3000],
            },
            {
                "name": "warm_neptune",
                "description": "Warm Neptune with moderate temperature",
                "default_molecules": ["H2O", "CH4", "CO2", "NH3"],
                "temperature_range": [500, 1000],
            },
            {
                "name": "super_earth",
                "description": "Super-Earth with rocky/ocean surface",
                "default_molecules": ["H2O", "CO2", "O3"],
                "temperature_range": [250, 400],
            },
            {
                "name": "earth_like",
                "description": "Earth-like planet in habitable zone",
                "default_molecules": ["H2O", "O2", "O3", "CO2", "CH4"],
                "temperature_range": [250, 320],
            },
        ],
        "available_molecules": [
            "H2O", "CO2", "CO", "CH4", "NH3", "O2", "O3",
            "Na", "K", "TiO", "VO", "FeH", "H2S", "HCN"
        ],
    }


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

@app.post("/api/v1/preprocess", response_model=PreprocessingResponse, tags=["Preprocessing"])
async def preprocess_spectrum(config: PreprocessingRequest):
    """
    Apply preprocessing pipeline to a spectrum.
    
    Steps:
    1. Outlier removal (sigma clipping)
    2. Smoothing (Savitzky-Golay, Gaussian, Median)
    3. Baseline correction (Polynomial, Spline, ALS)
    4. Normalization
    """
    import time
    start_time = time.time()
    
    # Get spectrum data
    if config.spectrum_id:
        spectrum = store.get_spectrum(config.spectrum_id)
        if not spectrum:
            raise HTTPException(status_code=404, detail="Spectrum not found")
        wavelengths = spectrum['wavelengths']
        flux = spectrum['flux']
    elif config.spectrum_data:
        wavelengths = np.array(config.spectrum_data.wavelengths)
        flux = np.array(config.spectrum_data.flux)
    else:
        raise HTTPException(status_code=400, detail="Must provide spectrum_id or spectrum_data")
    
    # Apply preprocessing
    result = apply_preprocessing(wavelengths, flux, config)
    
    # Save processed spectrum
    new_id = store.save_spectrum(
        wavelengths=result['wavelengths'],
        flux=result['flux'],
        metadata={
            'preprocessed': True,
            'original_id': config.spectrum_id,
            'steps': result['steps'],
        }
    )
    
    processing_time = (time.time() - start_time) * 1000
    
    return PreprocessingResponse(
        spectrum_id=new_id,
        original_points=len(flux),
        processed_points=len(result['flux']),
        outliers_removed=result['outliers_removed'],
        processing_steps=result['steps'],
        wavelengths=result['wavelengths'].tolist(),
        flux=result['flux'].tolist(),
        processing_time_ms=processing_time,
    )


@app.get("/api/v1/preprocess/methods", tags=["Preprocessing"])
async def list_preprocessing_methods():
    """List available preprocessing methods and parameters."""
    return {
        "smoothing_methods": [
            {"name": "savgol", "description": "Savitzky-Golay filter", "params": ["window", "order"]},
            {"name": "gaussian", "description": "Gaussian smoothing", "params": ["sigma"]},
            {"name": "median", "description": "Median filter", "params": ["window"]},
            {"name": "lowess", "description": "LOWESS smoothing", "params": ["frac"]},
        ],
        "baseline_methods": [
            {"name": "polynomial", "description": "Polynomial fit", "params": ["order"]},
            {"name": "spline", "description": "Spline interpolation", "params": ["knots"]},
            {"name": "als", "description": "Asymmetric Least Squares", "params": ["lam", "p"]},
        ],
        "normalization_methods": ["minmax", "zscore", "median", "continuum"],
    }


# =============================================================================
# ML Inference
# =============================================================================

@app.post("/api/v1/inference", response_model=PredictionResponse, tags=["Inference"])
async def run_model_inference(request: InferenceRequest):
    """
    Run ML model inference on spectrum.
    
    Returns:
    - Detected molecules with probabilities
    - Planet classification
    - Habitability score
    - Uncertainty estimates (optional)
    """
    import time
    start_time = time.time()
    
    # Get spectrum
    if request.spectrum_id:
        spectrum = store.get_spectrum(request.spectrum_id)
        if not spectrum:
            raise HTTPException(status_code=404, detail="Spectrum not found")
        wavelengths = spectrum['wavelengths']
        flux = spectrum['flux']
        spectrum_id = request.spectrum_id
    elif request.spectrum_data:
        wavelengths = np.array(request.spectrum_data.wavelengths)
        flux = np.array(request.spectrum_data.flux)
        spectrum_id = store.save_spectrum(wavelengths, flux)
    else:
        raise HTTPException(status_code=400, detail="Must provide spectrum_id or spectrum_data")
    
    # Optional preprocessing
    if request.preprocess:
        preprocess_config = PreprocessingRequest(
            smoothing_method=PreprocessingMethod.SAVGOL,
            smoothing_window=11,
            remove_outliers=True,
            normalize=True,
        )
        result = apply_preprocessing(wavelengths, flux, preprocess_config)
        flux = result['flux']
    
    # Run inference
    prediction = run_inference(
        wavelengths, flux,
        request.model_type,
        request.with_uncertainty,
    )
    
    # Store prediction
    store.save_prediction(spectrum_id, prediction)
    
    inference_time = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        spectrum_id=spectrum_id,
        model_type=request.model_type.value,
        detected_molecules=prediction['detected_molecules'],
        molecule_probabilities=prediction['molecule_probabilities'],
        molecule_uncertainties=prediction.get('molecule_uncertainties'),
        planet_class=prediction['planet_class'],
        planet_probabilities=prediction['planet_probabilities'],
        planet_uncertainty=prediction.get('planet_uncertainty'),
        habitability_score=prediction['habitability_score'],
        habitability_uncertainty=prediction.get('habitability_uncertainty'),
        inference_time_ms=inference_time,
        model_version=prediction['model_version'],
    )


@app.post("/api/v1/inference/batch", tags=["Inference"])
async def run_batch_inference(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
):
    """Run inference on multiple spectra."""
    results = []
    
    for spectrum_id in request.spectrum_ids:
        spectrum = store.get_spectrum(spectrum_id)
        if not spectrum:
            results.append({'spectrum_id': spectrum_id, 'error': 'Not found'})
            continue
        
        prediction = run_inference(
            spectrum['wavelengths'],
            spectrum['flux'],
            request.model_type,
            request.with_uncertainty,
        )
        
        store.save_prediction(spectrum_id, prediction)
        
        results.append({
            'spectrum_id': spectrum_id,
            'detected_molecules': prediction['detected_molecules'],
            'planet_class': prediction['planet_class'],
            'habitability_score': prediction['habitability_score'],
        })
    
    return {
        'batch_size': len(request.spectrum_ids),
        'results': results,
    }


@app.get("/api/v1/inference/models", tags=["Inference"])
async def list_models():
    """List available ML models."""
    return {
        "models": [
            {
                "type": "cnn",
                "name": "ExoplanetSpectrumCNN",
                "version": "1.0",
                "description": "1D CNN with residual blocks for spectral classification",
                "input_length": 512,
                "tasks": ["molecule_detection", "planet_classification", "habitability"],
            },
            {
                "type": "transformer",
                "name": "SpectralTransformer",
                "version": "1.0",
                "description": "Transformer with wavelength-aware positional encoding",
                "input_length": 512,
                "tasks": ["molecule_detection", "planet_classification", "habitability"],
                "features": ["uncertainty_estimation", "attention_visualization"],
            },
        ],
    }


# =============================================================================
# Explainability
# =============================================================================

@app.post("/api/v1/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_prediction(request: ExplainRequest):
    """
    Generate explainability outputs for a prediction.
    
    Methods:
    - Gradient saliency
    - SmoothGrad
    - Integrated Gradients
    - Occlusion analysis
    
    Returns:
    - Saliency map (wavelength importance)
    - Influential absorption regions
    - Molecule-specific attributions
    """
    import time
    start_time = time.time()
    
    # Get spectrum
    if request.spectrum_id:
        spectrum = store.get_spectrum(request.spectrum_id)
        if not spectrum:
            raise HTTPException(status_code=404, detail="Spectrum not found")
        wavelengths = spectrum['wavelengths']
        flux = spectrum['flux']
        spectrum_id = request.spectrum_id
    elif request.spectrum_data:
        wavelengths = np.array(request.spectrum_data.wavelengths)
        flux = np.array(request.spectrum_data.flux)
        spectrum_id = store.save_spectrum(wavelengths, flux)
    else:
        raise HTTPException(status_code=400, detail="Must provide spectrum_id or spectrum_data")
    
    # Compute saliency
    result = compute_saliency(wavelengths, flux, request.method, request.target_molecule)
    
    # Store explanation
    store.save_explanation(spectrum_id, {
        'saliency': result['saliency'].tolist(),
        'regions': result['regions'],
        'method': request.method.value,
    })
    
    processing_time = (time.time() - start_time) * 1000
    
    return ExplainResponse(
        spectrum_id=spectrum_id,
        method=request.method.value,
        saliency=result['saliency'].tolist(),
        wavelengths=wavelengths.tolist(),
        regions=result['regions'],
        molecule_attributions=None,  # Would contain per-molecule saliencies
        processing_time_ms=processing_time,
    )


@app.get("/api/v1/explain/{spectrum_id}", tags=["Explainability"])
async def get_explanation(spectrum_id: str):
    """Retrieve stored explanation for a spectrum."""
    explanation = store.get_explanation(spectrum_id)
    
    if not explanation:
        raise HTTPException(status_code=404, detail="Explanation not found")
    
    return explanation


@app.get("/api/v1/explain/methods", tags=["Explainability"])
async def list_explain_methods():
    """List available explainability methods."""
    return {
        "methods": [
            {
                "name": "gradient",
                "description": "Basic gradient saliency",
                "speed": "fast",
                "quality": "low",
            },
            {
                "name": "smooth_grad",
                "description": "Averaged gradients over noisy inputs",
                "speed": "medium",
                "quality": "medium",
            },
            {
                "name": "integrated_gradients",
                "description": "Path-integrated attributions",
                "speed": "slow",
                "quality": "high",
            },
            {
                "name": "occlusion",
                "description": "Sliding window occlusion analysis",
                "speed": "slow",
                "quality": "high",
            },
        ],
    }


# =============================================================================
# Export Endpoints
# =============================================================================

@app.post("/api/v1/export", tags=["Export"])
async def export_results(request: ExportRequest):
    """
    Export analysis results to JSON or PDF.
    
    Includes:
    - Raw spectrum data
    - Predictions
    - Explanations
    - Visualization plots (PDF only)
    """
    spectrum = store.get_spectrum(request.spectrum_id)
    if not spectrum:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    
    if request.format == "json":
        export_data = export_to_json(request.spectrum_id)
        
        # Save to file
        filename = f"export_{request.spectrum_id}.json"
        filepath = RESULTS_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return FileResponse(
            filepath,
            media_type="application/json",
            filename=filename,
        )
    
    elif request.format == "pdf":
        pdf_bytes = export_to_pdf(request.spectrum_id)
        
        filename = f"report_{request.spectrum_id}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'pdf'")


@app.get("/api/v1/export/{spectrum_id}/json", tags=["Export"])
async def export_json(spectrum_id: str):
    """Quick export to JSON."""
    export_data = export_to_json(spectrum_id)
    return JSONResponse(content=export_data)


@app.get("/api/v1/export/{spectrum_id}/pdf", tags=["Export"])
async def export_pdf(spectrum_id: str):
    """Quick export to PDF."""
    pdf_bytes = export_to_pdf(spectrum_id)
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{spectrum_id}.pdf"},
    )


# =============================================================================
# Analysis Pipeline (Combined Endpoint)
# =============================================================================

@app.post("/api/v1/analyze", tags=["Analysis"])
async def full_analysis_pipeline(
    file: Optional[UploadFile] = File(None),
    spectrum_data: Optional[str] = None,
    model_type: ModelType = Query(ModelType.TRANSFORMER),
    preprocess: bool = Query(True),
    explain: bool = Query(True),
):
    """
    Run complete analysis pipeline:
    1. Upload/parse spectrum
    2. Preprocess
    3. Run inference
    4. Generate explanations
    
    Returns comprehensive analysis results.
    """
    import time
    start_time = time.time()
    
    # Step 1: Get spectrum
    if file:
        content = await file.read()
        filename = file.filename or "unknown"
        
        if filename.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            wl_col = next((c for c in df.columns if 'wave' in c.lower()), df.columns[0])
            flux_col = next((c for c in df.columns if 'flux' in c.lower()), df.columns[1])
            wavelengths = df[wl_col].values
            flux = df[flux_col].values
        elif filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            wavelengths = np.array(data['wavelengths'])
            flux = np.array(data['flux'])
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
    elif spectrum_data:
        data = json.loads(spectrum_data)
        wavelengths = np.array(data['wavelengths'])
        flux = np.array(data['flux'])
    else:
        raise HTTPException(status_code=400, detail="Must provide file or spectrum_data")
    
    spectrum_id = store.save_spectrum(wavelengths, flux)
    
    # Step 2: Preprocess
    preprocessing_result = None
    if preprocess:
        config = PreprocessingRequest(
            smoothing_method=PreprocessingMethod.SAVGOL,
            smoothing_window=11,
            remove_outliers=True,
            normalize=True,
        )
        preprocessing_result = apply_preprocessing(wavelengths, flux, config)
        flux = preprocessing_result['flux']
    
    # Step 3: Inference
    prediction = run_inference(wavelengths, flux, model_type, with_uncertainty=True)
    store.save_prediction(spectrum_id, prediction)
    
    # Step 4: Explainability
    explanation = None
    if explain:
        explanation = compute_saliency(
            wavelengths, flux,
            ExplainMethod.INTEGRATED_GRADIENTS,
        )
        store.save_explanation(spectrum_id, {
            'saliency': explanation['saliency'].tolist(),
            'regions': explanation['regions'],
        })
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        'spectrum_id': spectrum_id,
        'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
        'num_points': len(wavelengths),
        
        'preprocessing': {
            'applied': preprocess,
            'steps': preprocessing_result['steps'] if preprocessing_result else [],
        },
        
        'prediction': {
            'detected_molecules': prediction['detected_molecules'],
            'molecule_probabilities': prediction['molecule_probabilities'],
            'planet_class': prediction['planet_class'],
            'habitability_score': prediction['habitability_score'],
            'habitability_uncertainty': prediction.get('habitability_uncertainty'),
        },
        
        'explanation': {
            'included': explain,
            'influential_regions': explanation['regions'] if explanation else [],
        },
        
        'export_urls': {
            'json': f'/api/v1/export/{spectrum_id}/json',
            'pdf': f'/api/v1/export/{spectrum_id}/pdf',
        },
        
        'total_time_ms': total_time,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
