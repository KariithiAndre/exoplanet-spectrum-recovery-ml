"""
Absorption Line Detection Module

Detects and characterizes absorption features in exoplanet transmission spectra.
Uses peak/valley detection algorithms to identify spectral lines and extract:
- Line center wavelength
- Absorption depth
- Line width (FWHM)
- Integrated area
- Molecular band association

Returns structured feature vectors suitable for ML classification and analysis.

Author: Exoplanet Spectrum Recovery Project
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal, ndimage, optimize, interpolate
from scipy.stats import norm

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Molecular Band Database
# =============================================================================

class MoleculeType(Enum):
    """Known molecular species in exoplanet atmospheres."""
    H2O = "H2O"
    CO2 = "CO2"
    CO = "CO"
    CH4 = "CH4"
    NH3 = "NH3"
    O2 = "O2"
    O3 = "O3"
    Na = "Na"
    K = "K"
    TiO = "TiO"
    VO = "VO"
    FeH = "FeH"
    H2S = "H2S"
    HCN = "HCN"
    C2H2 = "C2H2"
    PH3 = "PH3"
    SiO = "SiO"
    UNKNOWN = "unknown"


@dataclass
class MolecularBand:
    """Definition of a molecular absorption band."""
    molecule: MoleculeType
    center_wavelength: float  # microns
    width: float  # microns (approximate band width)
    strength: str  # "weak", "medium", "strong"
    description: str = ""
    
    @property
    def wavelength_range(self) -> Tuple[float, float]:
        """Get wavelength range of the band."""
        half_width = self.width / 2
        return (self.center_wavelength - half_width, self.center_wavelength + half_width)


# Comprehensive molecular band database (wavelengths in microns)
MOLECULAR_BANDS: List[MolecularBand] = [
    # Water (H2O) - ubiquitous in exoplanet atmospheres
    MolecularBand(MoleculeType.H2O, 0.72, 0.02, "weak", "H2O weak band"),
    MolecularBand(MoleculeType.H2O, 0.82, 0.03, "weak", "H2O 820nm band"),
    MolecularBand(MoleculeType.H2O, 0.94, 0.04, "medium", "H2O 940nm band"),
    MolecularBand(MoleculeType.H2O, 1.14, 0.06, "medium", "H2O 1.14μm band"),
    MolecularBand(MoleculeType.H2O, 1.38, 0.10, "strong", "H2O 1.4μm band"),
    MolecularBand(MoleculeType.H2O, 1.87, 0.12, "strong", "H2O 1.9μm band"),
    MolecularBand(MoleculeType.H2O, 2.70, 0.20, "strong", "H2O 2.7μm fundamental"),
    MolecularBand(MoleculeType.H2O, 5.50, 0.50, "strong", "H2O bending mode"),
    MolecularBand(MoleculeType.H2O, 6.27, 0.30, "strong", "H2O 6.3μm band"),
    
    # Carbon Dioxide (CO2)
    MolecularBand(MoleculeType.CO2, 1.43, 0.03, "weak", "CO2 1.43μm band"),
    MolecularBand(MoleculeType.CO2, 1.57, 0.03, "weak", "CO2 1.57μm band"),
    MolecularBand(MoleculeType.CO2, 1.96, 0.04, "medium", "CO2 1.96μm band"),
    MolecularBand(MoleculeType.CO2, 2.01, 0.04, "medium", "CO2 2.01μm band"),
    MolecularBand(MoleculeType.CO2, 2.69, 0.06, "medium", "CO2 2.7μm band"),
    MolecularBand(MoleculeType.CO2, 4.26, 0.15, "strong", "CO2 4.3μm fundamental"),
    MolecularBand(MoleculeType.CO2, 4.80, 0.10, "medium", "CO2 4.8μm band"),
    MolecularBand(MoleculeType.CO2, 15.0, 1.00, "strong", "CO2 15μm bending"),
    
    # Carbon Monoxide (CO)
    MolecularBand(MoleculeType.CO, 1.58, 0.04, "weak", "CO first overtone"),
    MolecularBand(MoleculeType.CO, 2.35, 0.08, "medium", "CO 2.3μm band"),
    MolecularBand(MoleculeType.CO, 4.67, 0.15, "strong", "CO fundamental"),
    
    # Methane (CH4)
    MolecularBand(MoleculeType.CH4, 0.89, 0.03, "weak", "CH4 890nm band"),
    MolecularBand(MoleculeType.CH4, 1.00, 0.03, "weak", "CH4 1.0μm band"),
    MolecularBand(MoleculeType.CH4, 1.16, 0.04, "weak", "CH4 1.16μm band"),
    MolecularBand(MoleculeType.CH4, 1.38, 0.05, "medium", "CH4 1.4μm band"),
    MolecularBand(MoleculeType.CH4, 1.66, 0.06, "medium", "CH4 1.7μm band"),
    MolecularBand(MoleculeType.CH4, 2.20, 0.08, "medium", "CH4 2.2μm band"),
    MolecularBand(MoleculeType.CH4, 2.37, 0.08, "medium", "CH4 2.4μm band"),
    MolecularBand(MoleculeType.CH4, 3.31, 0.15, "strong", "CH4 3.3μm fundamental"),
    MolecularBand(MoleculeType.CH4, 7.66, 0.30, "strong", "CH4 7.7μm band"),
    
    # Ammonia (NH3)
    MolecularBand(MoleculeType.NH3, 1.50, 0.05, "weak", "NH3 1.5μm band"),
    MolecularBand(MoleculeType.NH3, 1.97, 0.06, "medium", "NH3 2.0μm band"),
    MolecularBand(MoleculeType.NH3, 2.25, 0.08, "medium", "NH3 2.25μm band"),
    MolecularBand(MoleculeType.NH3, 2.90, 0.10, "medium", "NH3 2.9μm band"),
    MolecularBand(MoleculeType.NH3, 3.00, 0.10, "strong", "NH3 3.0μm band"),
    MolecularBand(MoleculeType.NH3, 6.15, 0.20, "strong", "NH3 umbrella mode"),
    MolecularBand(MoleculeType.NH3, 10.35, 0.40, "strong", "NH3 10.4μm band"),
    
    # Alkali Metals - Narrow atomic lines
    MolecularBand(MoleculeType.Na, 0.5890, 0.002, "strong", "Na D2 line"),
    MolecularBand(MoleculeType.Na, 0.5896, 0.002, "strong", "Na D1 line"),
    MolecularBand(MoleculeType.K, 0.7665, 0.002, "strong", "K 766.5nm line"),
    MolecularBand(MoleculeType.K, 0.7699, 0.002, "strong", "K 769.9nm line"),
    
    # Oxygen
    MolecularBand(MoleculeType.O2, 0.688, 0.01, "weak", "O2 B-band"),
    MolecularBand(MoleculeType.O2, 0.762, 0.01, "strong", "O2 A-band"),
    MolecularBand(MoleculeType.O2, 1.27, 0.02, "weak", "O2 1.27μm band"),
    
    # Ozone
    MolecularBand(MoleculeType.O3, 0.60, 0.05, "medium", "O3 Chappuis band"),
    MolecularBand(MoleculeType.O3, 9.60, 0.40, "strong", "O3 9.6μm band"),
    
    # Metal Oxides (hot Jupiters)
    MolecularBand(MoleculeType.TiO, 0.62, 0.02, "medium", "TiO gamma band"),
    MolecularBand(MoleculeType.TiO, 0.71, 0.03, "medium", "TiO epsilon band"),
    MolecularBand(MoleculeType.TiO, 0.77, 0.02, "medium", "TiO delta band"),
    MolecularBand(MoleculeType.TiO, 0.84, 0.02, "weak", "TiO phi band"),
    MolecularBand(MoleculeType.VO, 0.74, 0.02, "medium", "VO B-X band"),
    MolecularBand(MoleculeType.VO, 0.79, 0.02, "medium", "VO A-X band"),
    MolecularBand(MoleculeType.VO, 1.05, 0.03, "weak", "VO 1.05μm band"),
    
    # Iron Hydride
    MolecularBand(MoleculeType.FeH, 0.99, 0.02, "weak", "FeH Wing-Ford band"),
    MolecularBand(MoleculeType.FeH, 1.20, 0.03, "weak", "FeH 1.2μm band"),
    MolecularBand(MoleculeType.FeH, 1.60, 0.04, "weak", "FeH 1.6μm band"),
    
    # Hydrogen Sulfide
    MolecularBand(MoleculeType.H2S, 1.59, 0.04, "weak", "H2S 1.6μm band"),
    MolecularBand(MoleculeType.H2S, 2.00, 0.05, "weak", "H2S 2.0μm band"),
    MolecularBand(MoleculeType.H2S, 2.60, 0.08, "medium", "H2S 2.6μm band"),
    MolecularBand(MoleculeType.H2S, 3.80, 0.10, "medium", "H2S fundamental"),
    
    # Hydrogen Cyanide
    MolecularBand(MoleculeType.HCN, 1.53, 0.03, "weak", "HCN 1.5μm band"),
    MolecularBand(MoleculeType.HCN, 3.00, 0.10, "medium", "HCN 3.0μm band"),
    MolecularBand(MoleculeType.HCN, 7.00, 0.20, "medium", "HCN bending"),
    MolecularBand(MoleculeType.HCN, 14.0, 0.50, "strong", "HCN 14μm band"),
    
    # Acetylene
    MolecularBand(MoleculeType.C2H2, 1.53, 0.03, "weak", "C2H2 1.5μm band"),
    MolecularBand(MoleculeType.C2H2, 3.03, 0.08, "medium", "C2H2 3.0μm band"),
    MolecularBand(MoleculeType.C2H2, 7.50, 0.20, "medium", "C2H2 7.5μm band"),
    MolecularBand(MoleculeType.C2H2, 13.7, 0.40, "strong", "C2H2 13.7μm band"),
    
    # Phosphine
    MolecularBand(MoleculeType.PH3, 2.80, 0.10, "weak", "PH3 2.8μm band"),
    MolecularBand(MoleculeType.PH3, 4.30, 0.12, "medium", "PH3 4.3μm band"),
    MolecularBand(MoleculeType.PH3, 9.00, 0.30, "medium", "PH3 9.0μm band"),
    MolecularBand(MoleculeType.PH3, 10.1, 0.30, "medium", "PH3 10.1μm band"),
    
    # Silicon Monoxide
    MolecularBand(MoleculeType.SiO, 4.00, 0.15, "medium", "SiO fundamental"),
    MolecularBand(MoleculeType.SiO, 8.20, 0.30, "strong", "SiO 8.2μm band"),
]


def get_molecular_bands_in_range(
    wl_min: float,
    wl_max: float,
    molecules: Optional[List[MoleculeType]] = None,
) -> List[MolecularBand]:
    """
    Get molecular bands within a wavelength range.
    
    Args:
        wl_min: Minimum wavelength (microns)
        wl_max: Maximum wavelength (microns)
        molecules: Filter to specific molecules (None = all)
        
    Returns:
        List of MolecularBand objects
    """
    bands = []
    for band in MOLECULAR_BANDS:
        if molecules is not None and band.molecule not in molecules:
            continue
        
        band_min, band_max = band.wavelength_range
        # Check for overlap
        if band_max >= wl_min and band_min <= wl_max:
            bands.append(band)
    
    return bands


# =============================================================================
# Data Classes for Detection Results
# =============================================================================

@dataclass
class LineProfile(Enum):
    """Line profile types."""
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    UNKNOWN = "unknown"


@dataclass
class AbsorptionLine:
    """Detected absorption line with extracted properties."""
    
    # Position
    center_wavelength: float  # microns
    center_index: int  # index in original array
    
    # Depth
    depth: float  # fractional depth (0-1)
    depth_ppm: float  # depth in parts per million
    continuum_level: float  # local continuum level
    
    # Width
    fwhm: float  # Full Width at Half Maximum (microns)
    sigma: float  # Gaussian sigma equivalent (microns)
    
    # Area
    equivalent_width: float  # Equivalent width (microns)
    integrated_area: float  # Integrated area under the line
    
    # Quality metrics
    snr: float  # Signal-to-noise ratio of detection
    significance: float  # Detection significance (sigma)
    asymmetry: float  # Line asymmetry (-1 to 1)
    
    # Profile fit
    profile_type: str = "gaussian"
    fit_params: Dict[str, float] = field(default_factory=dict)
    fit_residual: float = 0.0
    
    # Molecular association
    molecule: Optional[MoleculeType] = None
    band_name: Optional[str] = None
    association_confidence: float = 0.0
    
    # Blend information
    is_blended: bool = False
    blend_components: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "center_wavelength": self.center_wavelength,
            "center_index": self.center_index,
            "depth": self.depth,
            "depth_ppm": self.depth_ppm,
            "continuum_level": self.continuum_level,
            "fwhm": self.fwhm,
            "sigma": self.sigma,
            "equivalent_width": self.equivalent_width,
            "integrated_area": self.integrated_area,
            "snr": self.snr,
            "significance": self.significance,
            "asymmetry": self.asymmetry,
            "profile_type": self.profile_type,
            "molecule": self.molecule.value if self.molecule else None,
            "band_name": self.band_name,
            "association_confidence": self.association_confidence,
            "is_blended": self.is_blended,
            "blend_components": self.blend_components,
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to feature vector for ML.
        
        Returns:
            1D numpy array of features
        """
        return np.array([
            self.center_wavelength,
            self.depth,
            self.depth_ppm / 1e6,  # Normalize
            self.fwhm,
            self.sigma,
            self.equivalent_width,
            self.integrated_area,
            self.snr,
            self.significance,
            self.asymmetry,
            float(self.is_blended),
            self.blend_components,
            self.association_confidence,
        ])


@dataclass
class DetectionResult:
    """Container for all detected absorption features."""
    
    lines: List[AbsorptionLine] = field(default_factory=list)
    
    # Detection metadata
    n_lines: int = 0
    wavelength_range: Tuple[float, float] = (0.0, 0.0)
    detection_threshold: float = 0.0
    
    # Molecular summary
    molecules_detected: List[MoleculeType] = field(default_factory=list)
    molecule_counts: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    mean_snr: float = 0.0
    detection_completeness: float = 0.0
    
    def __post_init__(self):
        self.n_lines = len(self.lines)
    
    def get_lines_by_molecule(self, molecule: MoleculeType) -> List[AbsorptionLine]:
        """Get all lines associated with a specific molecule."""
        return [line for line in self.lines if line.molecule == molecule]
    
    def get_strongest_lines(self, n: int = 10) -> List[AbsorptionLine]:
        """Get the n strongest lines by depth."""
        return sorted(self.lines, key=lambda x: x.depth, reverse=True)[:n]
    
    def to_feature_matrix(self) -> np.ndarray:
        """
        Convert all lines to feature matrix for ML.
        
        Returns:
            2D numpy array of shape (n_lines, n_features)
        """
        if not self.lines:
            return np.array([]).reshape(0, 13)
        
        return np.vstack([line.to_feature_vector() for line in self.lines])
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([line.to_dict() for line in self.lines])


# =============================================================================
# Line Profile Functions
# =============================================================================

def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian profile."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """Lorentzian profile."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float) -> np.ndarray:
    """
    Voigt profile (convolution of Gaussian and Lorentzian).
    Approximation using pseudo-Voigt.
    """
    # Mixing parameter
    f_L = gamma / (sigma + gamma)
    
    G = gaussian(x, 1.0, center, sigma)
    L = lorentzian(x, 1.0, center, gamma)
    
    return amplitude * (f_L * L + (1 - f_L) * G)


def fit_gaussian(
    wavelength: np.ndarray,
    flux: np.ndarray,
    center_guess: float,
    depth_guess: float,
    sigma_guess: float,
) -> Tuple[np.ndarray, float]:
    """
    Fit Gaussian profile to absorption line.
    
    Returns:
        Tuple of (fitted_params, residual)
    """
    try:
        # Fit inverted Gaussian for absorption
        def absorption_gaussian(x, amplitude, center, sigma, baseline):
            return baseline - gaussian(x, amplitude, center, sigma)
        
        p0 = [depth_guess, center_guess, sigma_guess, 1.0]
        bounds = (
            [0, wavelength.min(), 1e-6, 0],
            [1, wavelength.max(), wavelength.ptp(), 2]
        )
        
        popt, _ = optimize.curve_fit(
            absorption_gaussian, wavelength, flux,
            p0=p0, bounds=bounds, maxfev=1000
        )
        
        fitted = absorption_gaussian(wavelength, *popt)
        residual = np.sqrt(np.mean((flux - fitted)**2))
        
        return popt, residual
    
    except Exception:
        return np.array([depth_guess, center_guess, sigma_guess, 1.0]), np.inf


def fit_lorentzian(
    wavelength: np.ndarray,
    flux: np.ndarray,
    center_guess: float,
    depth_guess: float,
    gamma_guess: float,
) -> Tuple[np.ndarray, float]:
    """Fit Lorentzian profile to absorption line."""
    try:
        def absorption_lorentzian(x, amplitude, center, gamma, baseline):
            return baseline - lorentzian(x, amplitude, center, gamma)
        
        p0 = [depth_guess, center_guess, gamma_guess, 1.0]
        bounds = (
            [0, wavelength.min(), 1e-6, 0],
            [1, wavelength.max(), wavelength.ptp(), 2]
        )
        
        popt, _ = optimize.curve_fit(
            absorption_lorentzian, wavelength, flux,
            p0=p0, bounds=bounds, maxfev=1000
        )
        
        fitted = absorption_lorentzian(wavelength, *popt)
        residual = np.sqrt(np.mean((flux - fitted)**2))
        
        return popt, residual
    
    except Exception:
        return np.array([depth_guess, center_guess, gamma_guess, 1.0]), np.inf


# =============================================================================
# Peak/Valley Detection Functions
# =============================================================================

def find_absorption_valleys(
    wavelength: np.ndarray,
    flux: np.ndarray,
    prominence: float = 0.001,
    width: int = 3,
    height: Optional[float] = None,
    distance: int = 5,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Find absorption features (valleys) in spectrum.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array (normalized)
        prominence: Minimum prominence of valleys
        width: Minimum width of valleys (in samples)
        height: Maximum height (minimum flux value)
        distance: Minimum distance between valleys (in samples)
        
    Returns:
        Tuple of (valley_indices, properties_dict)
    """
    # Invert flux to find valleys as peaks
    inverted_flux = -flux
    
    # Find peaks in inverted spectrum
    peaks, properties = signal.find_peaks(
        inverted_flux,
        prominence=prominence,
        width=width,
        height=-height if height is not None else None,
        distance=distance,
    )
    
    # Add additional properties
    if len(peaks) > 0:
        properties['prominences'] = properties.get('prominences', np.zeros(len(peaks)))
        properties['widths'] = properties.get('widths', np.zeros(len(peaks)))
        properties['left_ips'] = properties.get('left_ips', np.zeros(len(peaks)))
        properties['right_ips'] = properties.get('right_ips', np.zeros(len(peaks)))
    
    return peaks, properties


def refine_line_center(
    wavelength: np.ndarray,
    flux: np.ndarray,
    peak_index: int,
    window: int = 5,
) -> Tuple[float, int]:
    """
    Refine line center using parabolic interpolation.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        peak_index: Initial peak index
        window: Half-window for fitting
        
    Returns:
        Tuple of (refined_wavelength, refined_index)
    """
    n = len(wavelength)
    left = max(0, peak_index - window)
    right = min(n, peak_index + window + 1)
    
    # Extract local region
    local_wl = wavelength[left:right]
    local_flux = flux[left:right]
    
    # Fit parabola
    if len(local_wl) >= 3:
        try:
            coeffs = np.polyfit(local_wl, local_flux, 2)
            # Minimum of parabola: x = -b/(2a)
            if coeffs[0] > 0:  # Concave up (absorption minimum)
                refined_wl = -coeffs[1] / (2 * coeffs[0])
                
                # Ensure within range
                if local_wl.min() <= refined_wl <= local_wl.max():
                    # Find closest index
                    refined_idx = np.argmin(np.abs(wavelength - refined_wl))
                    return refined_wl, refined_idx
        except Exception:
            pass
    
    return wavelength[peak_index], peak_index


def estimate_local_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    center_index: int,
    window: int = 20,
) -> float:
    """
    Estimate local continuum level around an absorption feature.
    
    Uses percentile-based estimation to avoid being biased by the absorption.
    """
    n = len(wavelength)
    left = max(0, center_index - window)
    right = min(n, center_index + window + 1)
    
    local_flux = flux[left:right]
    
    # Use upper percentile to estimate continuum
    continuum = np.percentile(local_flux, 85)
    
    return continuum


def measure_line_width(
    wavelength: np.ndarray,
    flux: np.ndarray,
    center_index: int,
    continuum: float,
    min_flux: float,
) -> Tuple[float, float, int, int]:
    """
    Measure FWHM and find line boundaries.
    
    Returns:
        Tuple of (fwhm, sigma, left_index, right_index)
    """
    depth = continuum - min_flux
    half_depth = continuum - depth / 2
    
    n = len(wavelength)
    
    # Find left boundary (where flux rises to half depth)
    left_idx = center_index
    for i in range(center_index, -1, -1):
        if flux[i] >= half_depth:
            left_idx = i
            break
    
    # Find right boundary
    right_idx = center_index
    for i in range(center_index, n):
        if flux[i] >= half_depth:
            right_idx = i
            break
    
    # Calculate FWHM
    fwhm = wavelength[right_idx] - wavelength[left_idx]
    
    # Convert to Gaussian sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    return fwhm, sigma, left_idx, right_idx


def calculate_equivalent_width(
    wavelength: np.ndarray,
    flux: np.ndarray,
    left_idx: int,
    right_idx: int,
    continuum: float,
) -> float:
    """
    Calculate equivalent width of absorption line.
    
    EW = integral of (1 - flux/continuum) dwl
    """
    if left_idx >= right_idx or continuum <= 0:
        return 0.0
    
    local_wl = wavelength[left_idx:right_idx + 1]
    local_flux = flux[left_idx:right_idx + 1]
    
    # Normalized flux
    normalized = local_flux / continuum
    
    # Integrate (1 - normalized)
    integrand = 1 - normalized
    ew = np.trapz(integrand, local_wl)
    
    return ew


def calculate_line_asymmetry(
    wavelength: np.ndarray,
    flux: np.ndarray,
    center_index: int,
    left_idx: int,
    right_idx: int,
) -> float:
    """
    Calculate line asymmetry.
    
    Returns value from -1 (blue-shifted) to +1 (red-shifted).
    """
    # Left wing area
    left_wl = wavelength[left_idx:center_index + 1]
    left_flux = flux[left_idx:center_index + 1]
    left_area = np.trapz(left_flux, left_wl) if len(left_wl) > 1 else 0
    
    # Right wing area
    right_wl = wavelength[center_index:right_idx + 1]
    right_flux = flux[center_index:right_idx + 1]
    right_area = np.trapz(right_flux, right_wl) if len(right_wl) > 1 else 0
    
    total = left_area + right_area
    if total == 0:
        return 0.0
    
    # Asymmetry: negative = left wing larger (blue-shifted absorption)
    asymmetry = (right_area - left_area) / total
    
    return np.clip(asymmetry, -1, 1)


# =============================================================================
# Molecular Association
# =============================================================================

def associate_with_molecule(
    center_wavelength: float,
    fwhm: float,
    tolerance_factor: float = 2.0,
) -> Tuple[Optional[MoleculeType], Optional[str], float]:
    """
    Associate detected line with known molecular band.
    
    Args:
        center_wavelength: Detected line center (microns)
        fwhm: Detected line width (microns)
        tolerance_factor: Matching tolerance (in units of band width)
        
    Returns:
        Tuple of (molecule, band_name, confidence)
    """
    best_match = None
    best_band_name = None
    best_confidence = 0.0
    
    for band in MOLECULAR_BANDS:
        band_min, band_max = band.wavelength_range
        
        # Check if line center is within band
        if band_min <= center_wavelength <= band_max:
            # Calculate match confidence based on distance from band center
            distance = abs(center_wavelength - band.center_wavelength)
            max_distance = band.width / 2 * tolerance_factor
            
            if distance <= max_distance:
                # Confidence decreases with distance from center
                confidence = 1.0 - (distance / max_distance)
                
                # Boost confidence for strong bands
                if band.strength == "strong":
                    confidence *= 1.2
                elif band.strength == "weak":
                    confidence *= 0.8
                
                # Check width compatibility
                width_ratio = fwhm / band.width if band.width > 0 else 1.0
                if 0.2 < width_ratio < 5.0:
                    confidence *= 1.0
                else:
                    confidence *= 0.5
                
                confidence = np.clip(confidence, 0, 1)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = band.molecule
                    best_band_name = band.description
    
    return best_match, best_band_name, best_confidence


# =============================================================================
# Main Detector Class
# =============================================================================

@dataclass
class DetectorConfig:
    """Configuration for line detection."""
    
    # Detection sensitivity
    prominence_threshold: float = 0.001  # Minimum prominence
    min_depth: float = 0.0005  # Minimum depth for detection
    min_width: int = 3  # Minimum width in samples
    min_distance: int = 5  # Minimum distance between lines
    
    # SNR requirements
    min_snr: float = 3.0  # Minimum SNR for valid detection
    min_significance: float = 3.0  # Minimum significance (sigma)
    
    # Profile fitting
    fit_profiles: bool = True
    profile_type: str = "gaussian"  # "gaussian", "lorentzian", "voigt", "auto"
    
    # Molecular association
    associate_molecules: bool = True
    association_tolerance: float = 2.0
    
    # Blending detection
    detect_blends: bool = True
    blend_separation: float = 0.01  # Minimum separation for non-blended lines (microns)
    
    # Continuum estimation
    continuum_window: int = 30  # Window for continuum estimation
    
    # Smoothing before detection
    smooth_before_detect: bool = True
    smooth_window: int = 5


class AbsorptionLineDetector:
    """
    Main class for detecting and characterizing absorption lines.
    
    Example:
        detector = AbsorptionLineDetector()
        result = detector.detect(wavelength, flux, error)
        
        for line in result.lines:
            print(f"Line at {line.center_wavelength:.4f} μm: "
                  f"depth={line.depth_ppm:.0f} ppm, "
                  f"molecule={line.molecule}")
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration."""
        self.config = config or DetectorConfig()
    
    def detect(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        error: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """
        Detect absorption lines in spectrum.
        
        Args:
            wavelength: Wavelength array (microns)
            flux: Flux array (normalized)
            error: Error array (optional)
            
        Returns:
            DetectionResult with all detected lines
        """
        # Validate input
        if len(wavelength) != len(flux):
            raise ValueError("Wavelength and flux arrays must have same length")
        
        if len(wavelength) < 10:
            return DetectionResult(
                wavelength_range=(wavelength.min(), wavelength.max()),
                detection_threshold=self.config.prominence_threshold,
            )
        
        # Optional smoothing
        if self.config.smooth_before_detect:
            detection_flux = ndimage.median_filter(flux, size=self.config.smooth_window)
        else:
            detection_flux = flux
        
        # Find valleys
        valleys, properties = find_absorption_valleys(
            wavelength,
            detection_flux,
            prominence=self.config.prominence_threshold,
            width=self.config.min_width,
            distance=self.config.min_distance,
        )
        
        if len(valleys) == 0:
            return DetectionResult(
                wavelength_range=(wavelength.min(), wavelength.max()),
                detection_threshold=self.config.prominence_threshold,
            )
        
        # Estimate noise level
        if error is not None:
            noise_level = np.median(error)
        else:
            # Estimate from high-frequency variations
            diff = np.diff(flux)
            noise_level = np.median(np.abs(diff)) / np.sqrt(2) * 1.4826
        
        # Process each detected valley
        lines = []
        
        for i, valley_idx in enumerate(valleys):
            # Refine center
            center_wl, center_idx = refine_line_center(
                wavelength, flux, valley_idx
            )
            
            # Estimate continuum
            continuum = estimate_local_continuum(
                wavelength, flux, center_idx, self.config.continuum_window
            )
            
            # Measure depth
            min_flux = flux[center_idx]
            depth = continuum - min_flux
            depth_ppm = depth * 1e6
            
            # Skip if below minimum depth
            if depth < self.config.min_depth:
                continue
            
            # Measure width
            fwhm, sigma, left_idx, right_idx = measure_line_width(
                wavelength, flux, center_idx, continuum, min_flux
            )
            
            # Calculate equivalent width and area
            ew = calculate_equivalent_width(
                wavelength, flux, left_idx, right_idx, continuum
            )
            
            # Integrated area
            local_wl = wavelength[left_idx:right_idx + 1]
            local_flux = flux[left_idx:right_idx + 1]
            integrated_area = np.trapz(continuum - local_flux, local_wl)
            
            # Calculate SNR and significance
            snr = depth / noise_level if noise_level > 0 else 0
            significance = properties['prominences'][i] / noise_level if noise_level > 0 else 0
            
            # Skip low SNR detections
            if snr < self.config.min_snr:
                continue
            
            # Calculate asymmetry
            asymmetry = calculate_line_asymmetry(
                wavelength, flux, center_idx, left_idx, right_idx
            )
            
            # Fit profile
            fit_params = {}
            fit_residual = 0.0
            profile_type = "gaussian"
            
            if self.config.fit_profiles:
                local_wl = wavelength[left_idx:right_idx + 1]
                local_flux = flux[left_idx:right_idx + 1]
                
                if len(local_wl) >= 4:
                    if self.config.profile_type in ["gaussian", "auto"]:
                        params, residual = fit_gaussian(
                            local_wl, local_flux, center_wl, depth, sigma
                        )
                        fit_params = {
                            "amplitude": params[0],
                            "center": params[1],
                            "sigma": params[2],
                            "baseline": params[3],
                        }
                        fit_residual = residual
                        profile_type = "gaussian"
            
            # Associate with molecule
            molecule = None
            band_name = None
            association_confidence = 0.0
            
            if self.config.associate_molecules:
                molecule, band_name, association_confidence = associate_with_molecule(
                    center_wl, fwhm, self.config.association_tolerance
                )
            
            # Create absorption line object
            line = AbsorptionLine(
                center_wavelength=center_wl,
                center_index=center_idx,
                depth=depth,
                depth_ppm=depth_ppm,
                continuum_level=continuum,
                fwhm=fwhm,
                sigma=sigma,
                equivalent_width=ew,
                integrated_area=integrated_area,
                snr=snr,
                significance=significance,
                asymmetry=asymmetry,
                profile_type=profile_type,
                fit_params=fit_params,
                fit_residual=fit_residual,
                molecule=molecule,
                band_name=band_name,
                association_confidence=association_confidence,
            )
            
            lines.append(line)
        
        # Detect blended lines
        if self.config.detect_blends and len(lines) > 1:
            lines = self._mark_blends(lines)
        
        # Build result
        result = DetectionResult(
            lines=lines,
            wavelength_range=(wavelength.min(), wavelength.max()),
            detection_threshold=self.config.prominence_threshold,
        )
        
        # Summarize molecules
        molecule_set = set()
        molecule_counts = {}
        
        for line in lines:
            if line.molecule is not None:
                molecule_set.add(line.molecule)
                mol_name = line.molecule.value
                molecule_counts[mol_name] = molecule_counts.get(mol_name, 0) + 1
        
        result.molecules_detected = list(molecule_set)
        result.molecule_counts = molecule_counts
        
        # Quality metrics
        if lines:
            result.mean_snr = np.mean([line.snr for line in lines])
        
        result.n_lines = len(lines)
        
        return result
    
    def _mark_blends(self, lines: List[AbsorptionLine]) -> List[AbsorptionLine]:
        """Mark blended lines based on separation."""
        # Sort by wavelength
        sorted_lines = sorted(lines, key=lambda x: x.center_wavelength)
        
        for i, line in enumerate(sorted_lines):
            # Check neighbors
            blend_count = 1
            
            if i > 0:
                prev_line = sorted_lines[i - 1]
                separation = line.center_wavelength - prev_line.center_wavelength
                if separation < self.config.blend_separation:
                    blend_count += 1
            
            if i < len(sorted_lines) - 1:
                next_line = sorted_lines[i + 1]
                separation = next_line.center_wavelength - line.center_wavelength
                if separation < self.config.blend_separation:
                    blend_count += 1
            
            line.is_blended = blend_count > 1
            line.blend_components = blend_count
        
        return sorted_lines


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_absorption_lines(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
    **config_kwargs,
) -> DetectionResult:
    """
    Convenience function for line detection.
    
    Args:
        wavelength: Wavelength array (microns)
        flux: Flux array
        error: Error array (optional)
        **config_kwargs: Configuration options
        
    Returns:
        DetectionResult with detected lines
    """
    config = DetectorConfig(**config_kwargs)
    detector = AbsorptionLineDetector(config)
    return detector.detect(wavelength, flux, error)


def get_feature_vectors(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
    max_features: int = 50,
) -> Tuple[np.ndarray, List[str]]:
    """
    Detect lines and return feature matrix for ML.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        error: Error array
        max_features: Maximum number of features to return
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    result = detect_absorption_lines(wavelength, flux, error)
    
    # Get strongest lines
    lines = result.get_strongest_lines(max_features)
    
    feature_names = [
        "center_wavelength",
        "depth",
        "depth_ppm_normalized",
        "fwhm",
        "sigma",
        "equivalent_width",
        "integrated_area",
        "snr",
        "significance",
        "asymmetry",
        "is_blended",
        "blend_components",
        "association_confidence",
    ]
    
    if not lines:
        return np.zeros((0, len(feature_names))), feature_names
    
    feature_matrix = np.vstack([line.to_feature_vector() for line in lines])
    
    return feature_matrix, feature_names


def summarize_molecular_content(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Detect lines and summarize molecular content.
    
    Returns summary suitable for classification tasks.
    """
    result = detect_absorption_lines(wavelength, flux, error)
    
    summary = {
        "n_lines_detected": result.n_lines,
        "molecules_detected": [m.value for m in result.molecules_detected],
        "molecule_counts": result.molecule_counts,
        "mean_snr": result.mean_snr,
        "wavelength_range": result.wavelength_range,
    }
    
    # Per-molecule statistics
    for molecule in MoleculeType:
        if molecule == MoleculeType.UNKNOWN:
            continue
        
        mol_lines = result.get_lines_by_molecule(molecule)
        
        if mol_lines:
            summary[f"{molecule.value}_detected"] = True
            summary[f"{molecule.value}_n_lines"] = len(mol_lines)
            summary[f"{molecule.value}_max_depth_ppm"] = max(l.depth_ppm for l in mol_lines)
            summary[f"{molecule.value}_total_ew"] = sum(l.equivalent_width for l in mol_lines)
        else:
            summary[f"{molecule.value}_detected"] = False
            summary[f"{molecule.value}_n_lines"] = 0
            summary[f"{molecule.value}_max_depth_ppm"] = 0.0
            summary[f"{molecule.value}_total_ew"] = 0.0
    
    return summary


# =============================================================================
# Main (Demo)
# =============================================================================

if __name__ == "__main__":
    # Generate synthetic spectrum with absorption features
    np.random.seed(42)
    
    n_points = 1000
    wavelength = np.linspace(0.6, 5.0, n_points)
    
    # Baseline
    flux = np.ones(n_points)
    
    # Add known molecular features
    features = [
        # (center, sigma, depth, molecule)
        (1.38, 0.05, 0.02, "H2O"),
        (1.87, 0.06, 0.025, "H2O"),
        (2.70, 0.10, 0.03, "H2O"),
        (2.35, 0.04, 0.015, "CO"),
        (4.26, 0.08, 0.035, "CO2"),
        (3.31, 0.07, 0.02, "CH4"),
        (0.589, 0.002, 0.01, "Na"),
        (0.766, 0.002, 0.008, "K"),
    ]
    
    for center, sigma, depth, mol in features:
        flux -= depth * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    
    # Add noise
    noise_level = 0.002
    flux += np.random.normal(0, noise_level, n_points)
    error = np.full(n_points, noise_level)
    
    print("=" * 70)
    print("Absorption Line Detection Demo")
    print("=" * 70)
    print(f"Spectrum: {n_points} points, {wavelength.min():.2f}-{wavelength.max():.2f} μm")
    print(f"Added {len(features)} synthetic features")
    
    # Detect lines
    detector = AbsorptionLineDetector()
    result = detector.detect(wavelength, flux, error)
    
    print(f"\nDetected {result.n_lines} absorption lines:")
    print("-" * 70)
    print(f"{'Center (μm)':<12} {'Depth (ppm)':<12} {'FWHM (μm)':<10} {'SNR':<8} {'Molecule':<10}")
    print("-" * 70)
    
    for line in result.get_strongest_lines(15):
        mol_str = line.molecule.value if line.molecule else "unknown"
        print(f"{line.center_wavelength:<12.4f} {line.depth_ppm:<12.0f} "
              f"{line.fwhm:<10.4f} {line.snr:<8.1f} {mol_str:<10}")
    
    print(f"\nMolecules detected: {[m.value for m in result.molecules_detected]}")
    print(f"Molecule counts: {result.molecule_counts}")
    
    # Get feature matrix
    features, names = get_feature_vectors(wavelength, flux, error)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature names: {names[:5]}...")
    
    # Get molecular summary
    summary = summarize_molecular_content(wavelength, flux, error)
    print(f"\nH2O detected: {summary['H2O_detected']}, lines: {summary['H2O_n_lines']}")
    print(f"CO2 detected: {summary['CO2_detected']}, lines: {summary['CO2_n_lines']}")
