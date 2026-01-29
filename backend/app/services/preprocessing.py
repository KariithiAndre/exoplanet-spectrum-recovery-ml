"""
Spectral Preprocessing Pipeline

Comprehensive preprocessing module for exoplanet transmission spectra.
Produces clean, ML-ready spectral data through a configurable pipeline.

Pipeline stages:
1. Wavelength normalization
2. Savitzky-Golay smoothing
3. Continuum baseline correction
4. Noise filtering (multiple methods)
5. Outlier removal (sigma clipping, MAD, isolation forest)

Author: Exoplanet Spectrum Recovery Project
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage, signal, interpolate
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation, zscore

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================

class SmoothingMethod(Enum):
    """Available smoothing methods."""
    SAVITZKY_GOLAY = "savgol"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    MOVING_AVERAGE = "moving_average"
    LOWESS = "lowess"
    WAVELET = "wavelet"


class BaselineCorrectionMethod(Enum):
    """Continuum/baseline correction methods."""
    POLYNOMIAL = "polynomial"
    SPLINE = "spline"
    ASYMMETRIC_LEAST_SQUARES = "als"
    MORPHOLOGICAL = "morphological"
    RUBBERBAND = "rubberband"
    ITERATIVE_POLYNOMIAL = "iterative_polynomial"


class NoiseFilterMethod(Enum):
    """Noise filtering methods."""
    BUTTERWORTH = "butterworth"
    WIENER = "wiener"
    WAVELET_DENOISE = "wavelet_denoise"
    MEDIAN_FILTER = "median_filter"
    BILATERAL = "bilateral"
    TOTAL_VARIATION = "total_variation"


class OutlierMethod(Enum):
    """Outlier detection/removal methods."""
    SIGMA_CLIP = "sigma_clip"
    MAD = "mad"  # Median Absolute Deviation
    IQR = "iqr"  # Interquartile Range
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    DBSCAN = "dbscan"
    ROLLING_MEDIAN = "rolling_median"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    
    # Wavelength normalization
    normalize_wavelength: bool = True
    wavelength_range: Tuple[float, float] = (0.0, 1.0)
    
    # Smoothing
    apply_smoothing: bool = True
    smoothing_method: SmoothingMethod = SmoothingMethod.SAVITZKY_GOLAY
    smoothing_window: int = 11
    smoothing_polyorder: int = 3
    smoothing_sigma: float = 2.0  # For Gaussian
    
    # Baseline correction
    apply_baseline_correction: bool = True
    baseline_method: BaselineCorrectionMethod = BaselineCorrectionMethod.POLYNOMIAL
    baseline_polynomial_degree: int = 3
    baseline_als_lambda: float = 1e6
    baseline_als_p: float = 0.01
    baseline_iterations: int = 10
    
    # Noise filtering
    apply_noise_filter: bool = True
    noise_filter_method: NoiseFilterMethod = NoiseFilterMethod.BUTTERWORTH
    noise_cutoff_frequency: float = 0.1
    noise_filter_order: int = 5
    
    # Outlier removal
    apply_outlier_removal: bool = True
    outlier_method: OutlierMethod = OutlierMethod.SIGMA_CLIP
    outlier_sigma: float = 3.0
    outlier_max_iterations: int = 5
    outlier_interpolate: bool = True
    
    # Error propagation
    propagate_errors: bool = True
    
    # Output options
    resample_to_uniform: bool = False
    resample_n_points: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class ProcessedSpectrum:
    """Container for processed spectrum data."""
    wavelength: np.ndarray
    flux: np.ndarray
    error: Optional[np.ndarray] = None
    
    # Original data
    original_wavelength: Optional[np.ndarray] = None
    original_flux: Optional[np.ndarray] = None
    original_error: Optional[np.ndarray] = None
    
    # Processing metadata
    processing_steps: List[str] = field(default_factory=list)
    processing_params: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    outliers_removed: int = 0
    snr_before: Optional[float] = None
    snr_after: Optional[float] = None
    
    # Mask for valid data
    valid_mask: Optional[np.ndarray] = None
    
    # Baseline that was removed
    baseline: Optional[np.ndarray] = None
    
    @property
    def n_points(self) -> int:
        return len(self.wavelength)
    
    @property
    def wavelength_range(self) -> Tuple[float, float]:
        return (float(self.wavelength.min()), float(self.wavelength.max()))
    
    def to_ml_array(self, include_error: bool = False) -> np.ndarray:
        """
        Convert to numpy array suitable for ML models.
        
        Returns:
            2D array of shape (n_points, 2) or (n_points, 3) with error
        """
        if include_error and self.error is not None:
            return np.column_stack([self.wavelength, self.flux, self.error])
        return np.column_stack([self.wavelength, self.flux])
    
    def to_torch_tensor(self, include_error: bool = False):
        """Convert to PyTorch tensor."""
        try:
            import torch
            arr = self.to_ml_array(include_error)
            return torch.from_numpy(arr).float()
        except ImportError:
            raise ImportError("PyTorch is required for tensor conversion")


# =============================================================================
# Preprocessing Functions
# =============================================================================

def normalize_wavelength(
    wavelength: np.ndarray,
    target_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize wavelength to a target range.
    
    Args:
        wavelength: Original wavelength array
        target_range: Target (min, max) range
        
    Returns:
        Tuple of (normalized_wavelength, normalization_params)
    """
    wl_min, wl_max = wavelength.min(), wavelength.max()
    target_min, target_max = target_range
    
    if wl_max == wl_min:
        normalized = np.full_like(wavelength, (target_min + target_max) / 2)
    else:
        normalized = (wavelength - wl_min) / (wl_max - wl_min)
        normalized = normalized * (target_max - target_min) + target_min
    
    params = {
        "original_min": float(wl_min),
        "original_max": float(wl_max),
        "target_min": target_min,
        "target_max": target_max,
    }
    
    return normalized, params


def denormalize_wavelength(
    normalized: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """Reverse wavelength normalization."""
    target_min = params["target_min"]
    target_max = params["target_max"]
    original_min = params["original_min"]
    original_max = params["original_max"]
    
    # Reverse normalization
    result = (normalized - target_min) / (target_max - target_min)
    result = result * (original_max - original_min) + original_min
    
    return result


# =============================================================================
# Smoothing Functions
# =============================================================================

def savitzky_golay_smooth(
    flux: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter.
    
    Args:
        flux: Input flux array
        window_length: Window size (must be odd)
        polyorder: Polynomial order for fitting
        deriv: Derivative order (0 for smoothing)
        
    Returns:
        Smoothed flux array
    """
    # Ensure window length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure window length > polyorder
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    
    # Handle edge case of small arrays
    if len(flux) < window_length:
        window_length = len(flux) if len(flux) % 2 == 1 else len(flux) - 1
        polyorder = min(polyorder, window_length - 1)
    
    return signal.savgol_filter(flux, window_length, polyorder, deriv=deriv)


def gaussian_smooth(
    flux: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Apply Gaussian smoothing."""
    return ndimage.gaussian_filter1d(flux, sigma)


def median_smooth(
    flux: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply median filter smoothing."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return ndimage.median_filter(flux, size=kernel_size)


def moving_average_smooth(
    flux: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Apply moving average smoothing."""
    kernel = np.ones(window) / window
    # Use 'same' mode and handle edges
    smoothed = np.convolve(flux, kernel, mode='same')
    return smoothed


def lowess_smooth(
    wavelength: np.ndarray,
    flux: np.ndarray,
    frac: float = 0.1,
) -> np.ndarray:
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing).
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        result = lowess(flux, wavelength, frac=frac, return_sorted=False)
        return result
    except ImportError:
        logger.warning("statsmodels not available, falling back to Savitzky-Golay")
        return savitzky_golay_smooth(flux)


def apply_smoothing(
    flux: np.ndarray,
    method: SmoothingMethod,
    wavelength: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply smoothing using specified method.
    
    Args:
        flux: Input flux array
        method: Smoothing method
        wavelength: Wavelength array (required for some methods)
        **kwargs: Method-specific parameters
        
    Returns:
        Smoothed flux array
    """
    if method == SmoothingMethod.SAVITZKY_GOLAY:
        window = kwargs.get("window_length", 11)
        polyorder = kwargs.get("polyorder", 3)
        return savitzky_golay_smooth(flux, window, polyorder)
    
    elif method == SmoothingMethod.GAUSSIAN:
        sigma = kwargs.get("sigma", 2.0)
        return gaussian_smooth(flux, sigma)
    
    elif method == SmoothingMethod.MEDIAN:
        kernel_size = kwargs.get("kernel_size", 5)
        return median_smooth(flux, kernel_size)
    
    elif method == SmoothingMethod.MOVING_AVERAGE:
        window = kwargs.get("window", 5)
        return moving_average_smooth(flux, window)
    
    elif method == SmoothingMethod.LOWESS:
        if wavelength is None:
            wavelength = np.arange(len(flux))
        frac = kwargs.get("frac", 0.1)
        return lowess_smooth(wavelength, flux, frac)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


# =============================================================================
# Baseline Correction Functions
# =============================================================================

def polynomial_baseline(
    wavelength: np.ndarray,
    flux: np.ndarray,
    degree: int = 3,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fit and return polynomial baseline.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        degree: Polynomial degree
        mask: Boolean mask for fitting (True = use for fit)
        
    Returns:
        Baseline array
    """
    if mask is None:
        mask = np.ones(len(flux), dtype=bool)
    
    valid = mask & np.isfinite(flux)
    
    if np.sum(valid) < degree + 1:
        return np.zeros_like(flux)
    
    coeffs = np.polyfit(wavelength[valid], flux[valid], degree)
    baseline = np.polyval(coeffs, wavelength)
    
    return baseline


def iterative_polynomial_baseline(
    wavelength: np.ndarray,
    flux: np.ndarray,
    degree: int = 3,
    n_iterations: int = 10,
    sigma_lower: float = 0.5,
    sigma_upper: float = 3.0,
) -> np.ndarray:
    """
    Iteratively fit polynomial baseline, excluding absorption features.
    
    Uses asymmetric sigma clipping to identify continuum points.
    """
    mask = np.ones(len(flux), dtype=bool) & np.isfinite(flux)
    
    for _ in range(n_iterations):
        if np.sum(mask) < degree + 1:
            break
            
        # Fit polynomial to masked points
        coeffs = np.polyfit(wavelength[mask], flux[mask], degree)
        baseline = np.polyval(coeffs, wavelength)
        
        # Calculate residuals
        residuals = flux - baseline
        std = np.std(residuals[mask])
        
        # Asymmetric clipping (more aggressive for negative residuals = absorption)
        mask = (
            (residuals > -sigma_lower * std) &
            (residuals < sigma_upper * std) &
            np.isfinite(flux)
        )
    
    return np.polyval(coeffs, wavelength)


def spline_baseline(
    wavelength: np.ndarray,
    flux: np.ndarray,
    n_knots: int = 10,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Fit spline baseline.
    """
    valid = np.isfinite(flux)
    
    if np.sum(valid) < 4:
        return np.zeros_like(flux)
    
    # Create knot positions
    wl_valid = wavelength[valid]
    knots = np.linspace(wl_valid.min(), wl_valid.max(), n_knots + 2)[1:-1]
    
    try:
        # Fit B-spline
        tck = interpolate.splrep(
            wavelength[valid], 
            flux[valid], 
            t=knots, 
            s=smoothing
        )
        baseline = interpolate.splev(wavelength, tck)
    except Exception:
        # Fallback to simple interpolation
        baseline = polynomial_baseline(wavelength, flux, degree=3)
    
    return baseline


def asymmetric_least_squares_baseline(
    flux: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    n_iterations: int = 10,
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.
    
    Reference: Eilers & Boelens (2005)
    
    Args:
        flux: Input spectrum
        lam: Smoothness parameter (larger = smoother)
        p: Asymmetry parameter (smaller = more asymmetric)
        n_iterations: Number of iterations
        
    Returns:
        Baseline array
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    L = len(flux)
    
    # Create second derivative matrix
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = D.T @ D
    
    # Initialize weights
    w = np.ones(L)
    
    for _ in range(n_iterations):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D
        baseline = spsolve(Z, w * flux)
        
        # Update weights asymmetrically
        w = p * (flux > baseline) + (1 - p) * (flux <= baseline)
    
    return baseline


def morphological_baseline(
    flux: np.ndarray,
    kernel_size: int = 50,
) -> np.ndarray:
    """
    Morphological baseline using opening operation.
    """
    # Erosion followed by dilation
    eroded = ndimage.grey_erosion(flux, size=kernel_size)
    baseline = ndimage.grey_dilation(eroded, size=kernel_size)
    
    return baseline


def rubberband_baseline(
    wavelength: np.ndarray,
    flux: np.ndarray,
) -> np.ndarray:
    """
    Rubberband baseline correction using convex hull.
    """
    from scipy.spatial import ConvexHull
    
    # Create points for convex hull
    points = np.column_stack([wavelength, flux])
    
    try:
        hull = ConvexHull(points)
        
        # Get vertices on the lower boundary
        vertices = hull.vertices
        
        # Sort vertices by x-coordinate
        sorted_vertices = sorted(vertices, key=lambda i: wavelength[i])
        
        # Find lower envelope
        lower_vertices = []
        for i, v in enumerate(sorted_vertices):
            if i == 0 or i == len(sorted_vertices) - 1:
                lower_vertices.append(v)
            elif flux[v] < flux[sorted_vertices[i-1]] or flux[v] < flux[sorted_vertices[i+1]]:
                lower_vertices.append(v)
        
        # Interpolate baseline
        if len(lower_vertices) >= 2:
            baseline = np.interp(
                wavelength,
                wavelength[lower_vertices],
                flux[lower_vertices]
            )
        else:
            baseline = np.min(flux) * np.ones_like(flux)
            
    except Exception:
        # Fallback
        baseline = np.min(flux) * np.ones_like(flux)
    
    return baseline


def apply_baseline_correction(
    wavelength: np.ndarray,
    flux: np.ndarray,
    method: BaselineCorrectionMethod,
    return_baseline: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply baseline correction using specified method.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        method: Baseline correction method
        return_baseline: If True, also return the baseline
        **kwargs: Method-specific parameters
        
    Returns:
        Corrected flux, or tuple of (corrected_flux, baseline)
    """
    if method == BaselineCorrectionMethod.POLYNOMIAL:
        degree = kwargs.get("degree", 3)
        baseline = polynomial_baseline(wavelength, flux, degree)
    
    elif method == BaselineCorrectionMethod.ITERATIVE_POLYNOMIAL:
        degree = kwargs.get("degree", 3)
        n_iter = kwargs.get("n_iterations", 10)
        baseline = iterative_polynomial_baseline(wavelength, flux, degree, n_iter)
    
    elif method == BaselineCorrectionMethod.SPLINE:
        n_knots = kwargs.get("n_knots", 10)
        baseline = spline_baseline(wavelength, flux, n_knots)
    
    elif method == BaselineCorrectionMethod.ASYMMETRIC_LEAST_SQUARES:
        lam = kwargs.get("lam", 1e6)
        p = kwargs.get("p", 0.01)
        n_iter = kwargs.get("n_iterations", 10)
        baseline = asymmetric_least_squares_baseline(flux, lam, p, n_iter)
    
    elif method == BaselineCorrectionMethod.MORPHOLOGICAL:
        kernel_size = kwargs.get("kernel_size", 50)
        baseline = morphological_baseline(flux, kernel_size)
    
    elif method == BaselineCorrectionMethod.RUBBERBAND:
        baseline = rubberband_baseline(wavelength, flux)
    
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    
    corrected = flux - baseline
    
    if return_baseline:
        return corrected, baseline
    return corrected


# =============================================================================
# Noise Filtering Functions
# =============================================================================

def butterworth_filter(
    flux: np.ndarray,
    cutoff: float = 0.1,
    order: int = 5,
    filter_type: str = "low",
) -> np.ndarray:
    """
    Apply Butterworth filter.
    
    Args:
        flux: Input flux
        cutoff: Cutoff frequency (0-1, normalized to Nyquist)
        order: Filter order
        filter_type: 'low' or 'high'
        
    Returns:
        Filtered flux
    """
    # Design filter
    b, a = signal.butter(order, cutoff, btype=filter_type)
    
    # Apply zero-phase filtering
    filtered = signal.filtfilt(b, a, flux)
    
    return filtered


def wiener_filter(
    flux: np.ndarray,
    noise_power: Optional[float] = None,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Apply Wiener filter for noise reduction.
    """
    return signal.wiener(flux, mysize=kernel_size, noise=noise_power)


def wavelet_denoise(
    flux: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """
    Apply wavelet denoising.
    """
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not available, falling back to Butterworth")
        return butterworth_filter(flux)
    
    # Decompose
    if level is None:
        level = min(pywt.dwt_max_level(len(flux), wavelet), 5)
    
    coeffs = pywt.wavedec(flux, wavelet, level=level)
    
    # Estimate noise from finest detail coefficients
    sigma = median_abs_deviation(coeffs[-1]) / 0.6745
    
    # Universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(flux)))
    
    # Threshold detail coefficients
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:
        if threshold_mode == "soft":
            denoised_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
        else:
            denoised_coeffs.append(pywt.threshold(c, threshold, mode="hard"))
    
    # Reconstruct
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    
    # Handle length mismatch
    if len(denoised) > len(flux):
        denoised = denoised[:len(flux)]
    elif len(denoised) < len(flux):
        denoised = np.pad(denoised, (0, len(flux) - len(denoised)), mode='edge')
    
    return denoised


def total_variation_denoise(
    flux: np.ndarray,
    weight: float = 0.1,
    n_iterations: int = 100,
) -> np.ndarray:
    """
    Total Variation denoising.
    
    Minimizes: ||y - x||^2 + weight * TV(x)
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
        # skimage expects 2D, so reshape
        flux_2d = flux.reshape(-1, 1)
        denoised_2d = denoise_tv_chambolle(flux_2d, weight=weight)
        return denoised_2d.flatten()
    except ImportError:
        # Simple gradient-based TV denoising
        denoised = flux.copy()
        for _ in range(n_iterations):
            # Gradient
            grad = np.diff(denoised, prepend=denoised[0])
            grad_backward = np.diff(denoised, append=denoised[-1])
            
            # Divergence of normalized gradient
            eps = 1e-8
            div = grad / (np.abs(grad) + eps) - grad_backward / (np.abs(grad_backward) + eps)
            
            # Update
            denoised = flux - weight * div
        
        return denoised


def apply_noise_filter(
    flux: np.ndarray,
    method: NoiseFilterMethod,
    **kwargs,
) -> np.ndarray:
    """
    Apply noise filtering using specified method.
    """
    if method == NoiseFilterMethod.BUTTERWORTH:
        cutoff = kwargs.get("cutoff", 0.1)
        order = kwargs.get("order", 5)
        return butterworth_filter(flux, cutoff, order)
    
    elif method == NoiseFilterMethod.WIENER:
        kernel_size = kwargs.get("kernel_size", 5)
        return wiener_filter(flux, kernel_size=kernel_size)
    
    elif method == NoiseFilterMethod.WAVELET_DENOISE:
        wavelet = kwargs.get("wavelet", "db4")
        return wavelet_denoise(flux, wavelet)
    
    elif method == NoiseFilterMethod.MEDIAN_FILTER:
        kernel_size = kwargs.get("kernel_size", 5)
        return median_smooth(flux, kernel_size)
    
    elif method == NoiseFilterMethod.TOTAL_VARIATION:
        weight = kwargs.get("weight", 0.1)
        return total_variation_denoise(flux, weight)
    
    else:
        raise ValueError(f"Unknown noise filter method: {method}")


# =============================================================================
# Outlier Removal Functions
# =============================================================================

def sigma_clip_outliers(
    flux: np.ndarray,
    sigma: float = 3.0,
    max_iterations: int = 5,
    center_func: Callable = np.nanmedian,
    std_func: Callable = np.nanstd,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outliers using sigma clipping.
    
    Returns:
        Tuple of (mask, cleaned_flux) where mask is True for valid points
    """
    mask = np.isfinite(flux)
    
    for _ in range(max_iterations):
        center = center_func(flux[mask])
        std = std_func(flux[mask])
        
        if std == 0:
            break
        
        new_mask = np.abs(flux - center) < sigma * std
        new_mask &= np.isfinite(flux)
        
        if np.array_equal(mask, new_mask):
            break
        
        mask = new_mask
    
    return mask, flux.copy()


def mad_outliers(
    flux: np.ndarray,
    threshold: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outliers using Median Absolute Deviation.
    
    More robust than sigma clipping for non-Gaussian distributions.
    """
    median = np.nanmedian(flux)
    mad = median_abs_deviation(flux, nan_policy='omit')
    
    if mad == 0:
        return np.isfinite(flux), flux.copy()
    
    # Modified Z-score
    modified_z = 0.6745 * (flux - median) / mad
    
    mask = np.abs(modified_z) < threshold
    mask &= np.isfinite(flux)
    
    return mask, flux.copy()


def iqr_outliers(
    flux: np.ndarray,
    factor: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outliers using Interquartile Range method.
    """
    q1 = np.nanpercentile(flux, 25)
    q3 = np.nanpercentile(flux, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    mask = (flux >= lower_bound) & (flux <= upper_bound) & np.isfinite(flux)
    
    return mask, flux.copy()


def rolling_median_outliers(
    flux: np.ndarray,
    window: int = 11,
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outliers using rolling median comparison.
    """
    # Calculate rolling median
    if window % 2 == 0:
        window += 1
    
    half_window = window // 2
    rolling_med = np.zeros_like(flux)
    
    for i in range(len(flux)):
        start = max(0, i - half_window)
        end = min(len(flux), i + half_window + 1)
        rolling_med[i] = np.nanmedian(flux[start:end])
    
    # Calculate residuals
    residuals = np.abs(flux - rolling_med)
    mad = median_abs_deviation(residuals, nan_policy='omit')
    
    if mad == 0:
        return np.isfinite(flux), flux.copy()
    
    mask = residuals < threshold * mad * 1.4826  # 1.4826 for consistency with std
    mask &= np.isfinite(flux)
    
    return mask, flux.copy()


def isolation_forest_outliers(
    flux: np.ndarray,
    contamination: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outliers using Isolation Forest.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        logger.warning("scikit-learn not available, falling back to sigma clipping")
        return sigma_clip_outliers(flux)
    
    # Reshape for sklearn
    X = flux.reshape(-1, 1)
    
    # Handle NaN
    valid = np.isfinite(flux)
    X_valid = X[valid]
    
    if len(X_valid) < 10:
        return valid, flux.copy()
    
    # Fit Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(X_valid)
    
    # Map back to full array
    mask = np.zeros(len(flux), dtype=bool)
    mask[valid] = predictions == 1
    
    return mask, flux.copy()


def interpolate_outliers(
    wavelength: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Interpolate to replace outlier values.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        mask: Boolean mask (True = valid, False = outlier)
        
    Returns:
        Flux array with outliers interpolated
    """
    if np.all(mask):
        return flux.copy()
    
    if not np.any(mask):
        return flux.copy()
    
    # Linear interpolation
    interpolated = np.interp(
        wavelength,
        wavelength[mask],
        flux[mask]
    )
    
    return interpolated


def apply_outlier_removal(
    wavelength: np.ndarray,
    flux: np.ndarray,
    method: OutlierMethod,
    interpolate: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply outlier removal using specified method.
    
    Returns:
        Tuple of (cleaned_flux, valid_mask, n_outliers_removed)
    """
    if method == OutlierMethod.SIGMA_CLIP:
        sigma = kwargs.get("sigma", 3.0)
        max_iter = kwargs.get("max_iterations", 5)
        mask, _ = sigma_clip_outliers(flux, sigma, max_iter)
    
    elif method == OutlierMethod.MAD:
        threshold = kwargs.get("threshold", 3.5)
        mask, _ = mad_outliers(flux, threshold)
    
    elif method == OutlierMethod.IQR:
        factor = kwargs.get("factor", 1.5)
        mask, _ = iqr_outliers(flux, factor)
    
    elif method == OutlierMethod.ROLLING_MEDIAN:
        window = kwargs.get("window", 11)
        threshold = kwargs.get("threshold", 3.0)
        mask, _ = rolling_median_outliers(flux, window, threshold)
    
    elif method == OutlierMethod.ISOLATION_FOREST:
        contamination = kwargs.get("contamination", 0.05)
        mask, _ = isolation_forest_outliers(flux, contamination)
    
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    
    n_outliers = np.sum(~mask)
    
    if interpolate and n_outliers > 0:
        cleaned_flux = interpolate_outliers(wavelength, flux, mask)
    else:
        cleaned_flux = flux.copy()
        cleaned_flux[~mask] = np.nan
    
    return cleaned_flux, mask, n_outliers


# =============================================================================
# Error Propagation
# =============================================================================

def propagate_error_through_smooth(
    error: np.ndarray,
    method: SmoothingMethod,
    **kwargs,
) -> np.ndarray:
    """
    Propagate errors through smoothing operations.
    
    For linear filters, errors combine in quadrature.
    """
    if method == SmoothingMethod.SAVITZKY_GOLAY:
        window = kwargs.get("window_length", 11)
        # Approximate error reduction
        return error / np.sqrt(window)
    
    elif method == SmoothingMethod.GAUSSIAN:
        sigma = kwargs.get("sigma", 2.0)
        effective_window = int(6 * sigma)
        return error / np.sqrt(effective_window)
    
    elif method == SmoothingMethod.MEDIAN:
        kernel_size = kwargs.get("kernel_size", 5)
        # Median has different statistics
        return error / np.sqrt(kernel_size * 0.64)  # π/2 efficiency factor
    
    elif method == SmoothingMethod.MOVING_AVERAGE:
        window = kwargs.get("window", 5)
        return error / np.sqrt(window)
    
    else:
        # Conservative: no reduction
        return error.copy()


def propagate_error_through_baseline(
    error: np.ndarray,
    method: BaselineCorrectionMethod,
    **kwargs,
) -> np.ndarray:
    """
    Propagate errors through baseline correction.
    
    Subtraction adds errors in quadrature; baseline fit has some uncertainty.
    """
    # Baseline fit uncertainty is typically small compared to data error
    # Add a small systematic component
    baseline_error_fraction = kwargs.get("baseline_error_fraction", 0.1)
    
    baseline_error = baseline_error_fraction * np.median(error)
    propagated = np.sqrt(error**2 + baseline_error**2)
    
    return propagated


# =============================================================================
# Resampling Functions
# =============================================================================

def resample_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
    n_points: int = 1000,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Resample spectrum to uniform wavelength grid.
    
    Args:
        wavelength: Original wavelength array
        flux: Flux array
        error: Error array (optional)
        n_points: Number of output points
        method: Interpolation method ('linear', 'cubic', 'spline')
        
    Returns:
        Tuple of (new_wavelength, new_flux, new_error)
    """
    # Create uniform grid
    new_wavelength = np.linspace(wavelength.min(), wavelength.max(), n_points)
    
    if method == "linear":
        new_flux = np.interp(new_wavelength, wavelength, flux)
    elif method == "cubic":
        f = interpolate.interp1d(wavelength, flux, kind='cubic', fill_value='extrapolate')
        new_flux = f(new_wavelength)
    elif method == "spline":
        tck = interpolate.splrep(wavelength, flux)
        new_flux = interpolate.splev(new_wavelength, tck)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    if error is not None:
        # Interpolate errors (approximate)
        new_error = np.interp(new_wavelength, wavelength, error)
        
        # Account for resampling factor
        resample_factor = len(wavelength) / n_points
        if resample_factor < 1:
            # Downsampling reduces error
            new_error *= np.sqrt(resample_factor)
    else:
        new_error = None
    
    return new_wavelength, new_flux, new_error


# =============================================================================
# Main Pipeline Class
# =============================================================================

class SpectralPreprocessor:
    """
    Main preprocessing pipeline for spectral data.
    
    Applies a configurable sequence of preprocessing steps:
    1. Wavelength normalization
    2. Outlier removal (first pass)
    3. Baseline correction
    4. Smoothing
    5. Noise filtering
    6. Outlier removal (second pass)
    7. Resampling (optional)
    
    Example:
        config = PreprocessingConfig(
            smoothing_method=SmoothingMethod.SAVITZKY_GOLAY,
            smoothing_window=15,
            baseline_method=BaselineCorrectionMethod.ITERATIVE_POLYNOMIAL,
            outlier_method=OutlierMethod.MAD,
        )
        
        preprocessor = SpectralPreprocessor(config)
        result = preprocessor.process(wavelength, flux, error)
        
        # Use result for ML
        ml_input = result.to_ml_array()
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or PreprocessingConfig()
    
    def process(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        error: Optional[np.ndarray] = None,
    ) -> ProcessedSpectrum:
        """
        Run the full preprocessing pipeline.
        
        Args:
            wavelength: Wavelength array
            flux: Flux array
            error: Error array (optional)
            
        Returns:
            ProcessedSpectrum with cleaned data and metadata
        """
        # Initialize result
        result = ProcessedSpectrum(
            wavelength=wavelength.copy(),
            flux=flux.copy(),
            error=error.copy() if error is not None else None,
            original_wavelength=wavelength.copy(),
            original_flux=flux.copy(),
            original_error=error.copy() if error is not None else None,
        )
        
        # Calculate initial SNR
        if error is not None:
            valid = error > 0
            if np.any(valid):
                result.snr_before = float(np.median(np.abs(flux[valid]) / error[valid]))
        
        # Track processing steps
        steps = []
        params = {}
        
        # 1. Wavelength normalization
        if self.config.normalize_wavelength:
            result.wavelength, wl_params = normalize_wavelength(
                result.wavelength,
                self.config.wavelength_range,
            )
            steps.append("wavelength_normalization")
            params["wavelength_normalization"] = wl_params
        
        # 2. First pass outlier removal
        if self.config.apply_outlier_removal:
            result.flux, mask, n_removed = apply_outlier_removal(
                result.wavelength,
                result.flux,
                self.config.outlier_method,
                interpolate=self.config.outlier_interpolate,
                sigma=self.config.outlier_sigma,
                max_iterations=self.config.outlier_max_iterations,
            )
            result.outliers_removed += n_removed
            result.valid_mask = mask
            steps.append("outlier_removal_pass1")
            params["outliers_removed_pass1"] = n_removed
        
        # 3. Baseline correction
        if self.config.apply_baseline_correction:
            result.flux, baseline = apply_baseline_correction(
                result.wavelength,
                result.flux,
                self.config.baseline_method,
                return_baseline=True,
                degree=self.config.baseline_polynomial_degree,
                lam=self.config.baseline_als_lambda,
                p=self.config.baseline_als_p,
                n_iterations=self.config.baseline_iterations,
            )
            result.baseline = baseline
            steps.append("baseline_correction")
            params["baseline_method"] = self.config.baseline_method.value
            
            if result.error is not None and self.config.propagate_errors:
                result.error = propagate_error_through_baseline(
                    result.error,
                    self.config.baseline_method,
                )
        
        # 4. Smoothing
        if self.config.apply_smoothing:
            result.flux = apply_smoothing(
                result.flux,
                self.config.smoothing_method,
                wavelength=result.wavelength,
                window_length=self.config.smoothing_window,
                polyorder=self.config.smoothing_polyorder,
                sigma=self.config.smoothing_sigma,
            )
            steps.append("smoothing")
            params["smoothing_method"] = self.config.smoothing_method.value
            params["smoothing_window"] = self.config.smoothing_window
            
            if result.error is not None and self.config.propagate_errors:
                result.error = propagate_error_through_smooth(
                    result.error,
                    self.config.smoothing_method,
                    window_length=self.config.smoothing_window,
                    sigma=self.config.smoothing_sigma,
                )
        
        # 5. Noise filtering
        if self.config.apply_noise_filter:
            result.flux = apply_noise_filter(
                result.flux,
                self.config.noise_filter_method,
                cutoff=self.config.noise_cutoff_frequency,
                order=self.config.noise_filter_order,
            )
            steps.append("noise_filtering")
            params["noise_filter_method"] = self.config.noise_filter_method.value
        
        # 6. Second pass outlier removal (after processing)
        if self.config.apply_outlier_removal:
            result.flux, mask, n_removed = apply_outlier_removal(
                result.wavelength,
                result.flux,
                self.config.outlier_method,
                interpolate=self.config.outlier_interpolate,
                sigma=self.config.outlier_sigma * 1.5,  # Less aggressive second pass
            )
            result.outliers_removed += n_removed
            if result.valid_mask is not None:
                result.valid_mask &= mask
            else:
                result.valid_mask = mask
            steps.append("outlier_removal_pass2")
            params["outliers_removed_pass2"] = n_removed
        
        # 7. Resampling
        if self.config.resample_to_uniform:
            result.wavelength, result.flux, result.error = resample_spectrum(
                result.wavelength,
                result.flux,
                result.error,
                n_points=self.config.resample_n_points,
            )
            steps.append("resampling")
            params["resample_n_points"] = self.config.resample_n_points
        
        # Calculate final SNR
        if result.error is not None:
            valid = result.error > 0
            if np.any(valid):
                result.snr_after = float(np.median(np.abs(result.flux[valid]) / result.error[valid]))
        
        result.processing_steps = steps
        result.processing_params = params
        
        return result
    
    def process_batch(
        self,
        spectra: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ) -> List[ProcessedSpectrum]:
        """
        Process multiple spectra.
        
        Args:
            spectra: List of (wavelength, flux, error) tuples
            
        Returns:
            List of ProcessedSpectrum objects
        """
        results = []
        
        for wavelength, flux, error in spectra:
            try:
                result = self.process(wavelength, flux, error)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process spectrum: {e}")
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def preprocess_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
    **config_kwargs,
) -> ProcessedSpectrum:
    """
    Convenience function for quick preprocessing.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array  
        error: Error array (optional)
        **config_kwargs: Configuration options
        
    Returns:
        ProcessedSpectrum object
    """
    config = PreprocessingConfig(**config_kwargs)
    preprocessor = SpectralPreprocessor(config)
    return preprocessor.process(wavelength, flux, error)


def quick_clean(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
) -> ProcessedSpectrum:
    """
    Quick preprocessing with sensible defaults.
    
    Applies: outlier removal, smoothing, and baseline correction.
    """
    config = PreprocessingConfig(
        normalize_wavelength=False,
        apply_smoothing=True,
        smoothing_method=SmoothingMethod.SAVITZKY_GOLAY,
        smoothing_window=7,
        apply_baseline_correction=True,
        baseline_method=BaselineCorrectionMethod.ITERATIVE_POLYNOMIAL,
        apply_noise_filter=False,
        apply_outlier_removal=True,
        outlier_method=OutlierMethod.MAD,
    )
    
    preprocessor = SpectralPreprocessor(config)
    return preprocessor.process(wavelength, flux, error)


def ml_ready_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
    n_points: int = 512,
) -> np.ndarray:
    """
    Preprocess spectrum and return ML-ready array.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        error: Error array (optional)
        n_points: Number of output points
        
    Returns:
        Numpy array of shape (n_points, 2) ready for ML input
    """
    config = PreprocessingConfig(
        normalize_wavelength=True,
        wavelength_range=(0.0, 1.0),
        apply_smoothing=True,
        apply_baseline_correction=True,
        apply_noise_filter=True,
        apply_outlier_removal=True,
        resample_to_uniform=True,
        resample_n_points=n_points,
    )
    
    preprocessor = SpectralPreprocessor(config)
    result = preprocessor.process(wavelength, flux, error)
    
    return result.to_ml_array()


# =============================================================================
# Main (Demo/Testing)
# =============================================================================

if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    
    n_points = 500
    wavelength = np.linspace(0.6, 5.0, n_points)  # JWST NIRSpec range
    
    # Create realistic spectrum with features
    true_signal = 1.0  # Baseline
    
    # Add absorption features
    for center, width, depth in [(1.4, 0.1, 0.02), (2.3, 0.15, 0.03), (4.3, 0.2, 0.025)]:
        true_signal -= depth * np.exp(-0.5 * ((wavelength - center) / width)**2)
    
    # Add baseline trend
    true_signal += 0.01 * (wavelength - wavelength.mean())
    
    # Add noise
    noise_level = 0.005
    noise = np.random.normal(0, noise_level, n_points)
    flux = true_signal + noise
    
    # Add outliers
    outlier_idx = np.random.choice(n_points, 10, replace=False)
    flux[outlier_idx] += np.random.uniform(-0.1, 0.1, 10)
    
    # Create error array
    error = np.full(n_points, noise_level)
    
    print("=" * 60)
    print("Spectral Preprocessing Pipeline Demo")
    print("=" * 60)
    print(f"Input: {n_points} points, wavelength {wavelength.min():.2f}-{wavelength.max():.2f} μm")
    print(f"Noise level: {noise_level}")
    print(f"Added {len(outlier_idx)} outliers")
    
    # Run preprocessing
    config = PreprocessingConfig(
        normalize_wavelength=True,
        smoothing_method=SmoothingMethod.SAVITZKY_GOLAY,
        smoothing_window=11,
        baseline_method=BaselineCorrectionMethod.ITERATIVE_POLYNOMIAL,
        baseline_polynomial_degree=2,
        noise_filter_method=NoiseFilterMethod.BUTTERWORTH,
        noise_cutoff_frequency=0.15,
        outlier_method=OutlierMethod.MAD,
    )
    
    preprocessor = SpectralPreprocessor(config)
    result = preprocessor.process(wavelength, flux, error)
    
    print(f"\nProcessing steps: {', '.join(result.processing_steps)}")
    print(f"Outliers removed: {result.outliers_removed}")
    print(f"SNR improvement: {result.snr_before:.1f} → {result.snr_after:.1f}")
    print(f"Output shape: {result.to_ml_array().shape}")
    
    # Quick clean demo
    print("\n--- Quick Clean ---")
    quick_result = quick_clean(wavelength, flux, error)
    print(f"Outliers removed: {quick_result.outliers_removed}")
    
    # ML-ready demo
    print("\n--- ML-Ready Output ---")
    ml_array = ml_ready_spectrum(wavelength, flux, error, n_points=256)
    print(f"ML array shape: {ml_array.shape}")
    print(f"Wavelength range (normalized): [{ml_array[:, 0].min():.3f}, {ml_array[:, 0].max():.3f}]")
