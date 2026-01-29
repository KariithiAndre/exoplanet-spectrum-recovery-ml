"""
Exoplanet Spectral Data Loader Module

This module provides comprehensive data loading capabilities for real exoplanet
spectral datasets from various sources including:
- JWST (James Webb Space Telescope) - NIRSpec, MIRI, NIRCam, NIRISS
- Hubble Space Telescope - WFC3, STIS, COS
- Ground-based facilities - VLT, Keck, Gemini
- NASA Exoplanet Archive format

Features:
- Automatic format detection
- Wavelength range validation
- Intensity normalization (multiple methods)
- Unit conversion
- Quality flag handling
- Metadata extraction
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class Instrument(Enum):
    """Supported telescope instruments."""
    # JWST Instruments
    JWST_NIRSPEC = "jwst_nirspec"
    JWST_MIRI = "jwst_miri"
    JWST_NIRCAM = "jwst_nircam"
    JWST_NIRISS = "jwst_niriss"
    
    # Hubble Instruments
    HST_WFC3 = "hst_wfc3"
    HST_STIS = "hst_stis"
    HST_COS = "hst_cos"
    
    # Ground-based
    VLT_XSHOOTER = "vlt_xshooter"
    KECK_NIRSPEC = "keck_nirspec"
    GEMINI_GNIRS = "gemini_gnirs"
    
    # Generic/Unknown
    GENERIC = "generic"
    UNKNOWN = "unknown"


class WavelengthUnit(Enum):
    """Wavelength unit types."""
    MICRON = "micron"
    NANOMETER = "nanometer"
    ANGSTROM = "angstrom"
    
    @classmethod
    def from_string(cls, unit_str: str) -> "WavelengthUnit":
        """Parse unit string to enum."""
        unit_str = unit_str.lower().strip()
        
        if unit_str in ["um", "μm", "micron", "microns", "micrometer"]:
            return cls.MICRON
        elif unit_str in ["nm", "nanometer", "nanometers"]:
            return cls.NANOMETER
        elif unit_str in ["a", "å", "angstrom", "angstroms"]:
            return cls.ANGSTROM
        else:
            raise ValueError(f"Unknown wavelength unit: {unit_str}")


class FluxUnit(Enum):
    """Flux/intensity unit types."""
    PPM = "ppm"                    # Parts per million (transit depth)
    PERCENT = "percent"            # Percentage
    FRACTION = "fraction"          # Fractional (0-1)
    FLUX_DENSITY = "flux_density"  # Jy or erg/s/cm²/Hz
    NORMALIZED = "normalized"      # Normalized to continuum
    RELATIVE = "relative"          # Relative units


class NormalizationMethod(Enum):
    """Normalization methods for intensity values."""
    NONE = "none"
    MIN_MAX = "min_max"            # Scale to [0, 1]
    ZSCORE = "zscore"              # (x - mean) / std
    MEDIAN = "median"              # Divide by median
    CONTINUUM = "continuum"        # Fit and divide by continuum
    PERCENTILE = "percentile"      # Scale using percentiles
    ROBUST = "robust"              # Robust scaling (IQR-based)


# Wavelength ranges for different instruments (in microns)
INSTRUMENT_WAVELENGTH_RANGES: Dict[Instrument, Tuple[float, float]] = {
    Instrument.JWST_NIRSPEC: (0.6, 5.3),
    Instrument.JWST_MIRI: (4.9, 28.8),
    Instrument.JWST_NIRCAM: (0.6, 5.0),
    Instrument.JWST_NIRISS: (0.6, 2.8),
    Instrument.HST_WFC3: (0.8, 1.7),
    Instrument.HST_STIS: (0.115, 1.03),
    Instrument.HST_COS: (0.09, 0.32),
    Instrument.VLT_XSHOOTER: (0.3, 2.5),
    Instrument.KECK_NIRSPEC: (0.95, 5.5),
    Instrument.GEMINI_GNIRS: (0.85, 5.4),
    Instrument.GENERIC: (0.1, 30.0),
    Instrument.UNKNOWN: (0.1, 30.0),
}


# Common column name mappings
COLUMN_MAPPINGS = {
    "wavelength": [
        "wavelength", "wave", "lambda", "wl", "wvl", "wavelength_um",
        "wavelength_micron", "wavelength_nm", "wave_um", "WAVELENGTH",
    ],
    "flux": [
        "flux", "intensity", "transit_depth", "depth", "spec", "spectrum",
        "flux_density", "transit_depth_ppm", "rp_rs_squared", "FLUX",
        "dppm", "depth_ppm", "td", "transit",
    ],
    "error": [
        "error", "err", "uncertainty", "sigma", "flux_err", "error_ppm",
        "unc", "e_flux", "flux_error", "FLUX_ERR", "dppm_err", "err_ppm",
    ],
    "quality": [
        "quality", "flag", "dq", "quality_flag", "mask", "good", "valid",
    ],
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpectrumMetadata:
    """Metadata for a loaded spectrum."""
    source_file: str
    instrument: Instrument = Instrument.UNKNOWN
    target_name: Optional[str] = None
    observation_date: Optional[str] = None
    exposure_time: Optional[float] = None
    
    wavelength_unit: WavelengthUnit = WavelengthUnit.MICRON
    flux_unit: FluxUnit = FluxUnit.PPM
    
    wavelength_range: Tuple[float, float] = (0.0, 0.0)
    n_points: int = 0
    
    telescope: Optional[str] = None
    program_id: Optional[str] = None
    pi_name: Optional[str] = None
    
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedSpectrum:
    """Container for a loaded and processed spectrum."""
    wavelength: np.ndarray
    flux: np.ndarray
    error: Optional[np.ndarray] = None
    quality_flags: Optional[np.ndarray] = None
    
    metadata: SpectrumMetadata = field(default_factory=lambda: SpectrumMetadata(""))
    
    # Original data (before normalization)
    original_flux: Optional[np.ndarray] = None
    original_error: Optional[np.ndarray] = None
    
    # Normalization info
    normalization_method: NormalizationMethod = NormalizationMethod.NONE
    normalization_params: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if len(self.wavelength) != len(self.flux):
            raise ValueError("Wavelength and flux arrays must have same length")
        
        if self.error is not None and len(self.error) != len(self.flux):
            raise ValueError("Error array must have same length as flux")
    
    @property
    def snr(self) -> Optional[float]:
        """Calculate signal-to-noise ratio."""
        if self.error is None:
            return None
        valid = self.error > 0
        if not np.any(valid):
            return None
        return float(np.median(np.abs(self.flux[valid]) / self.error[valid]))
    
    @property
    def wavelength_range(self) -> Tuple[float, float]:
        """Get wavelength range."""
        return (float(self.wavelength.min()), float(self.wavelength.max()))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            "wavelength": self.wavelength,
            "flux": self.flux,
        }
        if self.error is not None:
            data["error"] = self.error
        if self.quality_flags is not None:
            data["quality"] = self.quality_flags
        if self.original_flux is not None:
            data["original_flux"] = self.original_flux
        
        return pd.DataFrame(data)
    
    def get_valid_mask(self) -> np.ndarray:
        """Get mask for valid (non-NaN, non-flagged) data points."""
        valid = np.isfinite(self.flux)
        
        if self.error is not None:
            valid &= np.isfinite(self.error) & (self.error > 0)
        
        if self.quality_flags is not None:
            valid &= self.quality_flags == 0
        
        return valid


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)


# =============================================================================
# Wavelength Utilities
# =============================================================================

def convert_wavelength_to_microns(
    wavelength: np.ndarray,
    unit: WavelengthUnit,
) -> np.ndarray:
    """Convert wavelength array to microns."""
    if unit == WavelengthUnit.MICRON:
        return wavelength.copy()
    elif unit == WavelengthUnit.NANOMETER:
        return wavelength / 1000.0
    elif unit == WavelengthUnit.ANGSTROM:
        return wavelength / 10000.0
    else:
        raise ValueError(f"Unknown wavelength unit: {unit}")


def detect_wavelength_unit(wavelength: np.ndarray) -> WavelengthUnit:
    """Automatically detect wavelength unit from data range."""
    wl_min, wl_max = wavelength.min(), wavelength.max()
    
    # Angstroms: typical range 1000-100000
    if wl_max > 1000 and wl_min > 100:
        return WavelengthUnit.ANGSTROM
    
    # Nanometers: typical range 100-10000
    elif wl_max > 100 and wl_min > 10:
        return WavelengthUnit.NANOMETER
    
    # Microns: typical range 0.1-30
    else:
        return WavelengthUnit.MICRON


def validate_wavelength_range(
    wavelength: np.ndarray,
    instrument: Instrument,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate wavelength range for a given instrument.
    
    Args:
        wavelength: Wavelength array in microns
        instrument: Instrument type
        strict: If True, require exact match; if False, allow overlap
        
    Returns:
        ValidationResult with errors/warnings
    """
    result = ValidationResult(is_valid=True)
    
    wl_min, wl_max = wavelength.min(), wavelength.max()
    inst_min, inst_max = INSTRUMENT_WAVELENGTH_RANGES[instrument]
    
    # Check for completely out of range
    if wl_max < inst_min or wl_min > inst_max:
        result.add_error(
            f"Wavelength range [{wl_min:.3f}, {wl_max:.3f}] μm is completely "
            f"outside instrument range [{inst_min:.3f}, {inst_max:.3f}] μm"
        )
        return result
    
    # Check for partial overlap
    if wl_min < inst_min:
        msg = f"Wavelength minimum ({wl_min:.3f} μm) is below instrument range ({inst_min:.3f} μm)"
        if strict:
            result.add_error(msg)
        else:
            result.add_warning(msg)
    
    if wl_max > inst_max:
        msg = f"Wavelength maximum ({wl_max:.3f} μm) exceeds instrument range ({inst_max:.3f} μm)"
        if strict:
            result.add_error(msg)
        else:
            result.add_warning(msg)
    
    # Check for reasonable spacing
    dwavelength = np.diff(wavelength)
    if np.any(dwavelength <= 0):
        result.add_error("Wavelength array is not strictly increasing")
    
    # Check for large gaps
    median_spacing = np.median(dwavelength)
    large_gaps = dwavelength > 10 * median_spacing
    if np.any(large_gaps):
        n_gaps = np.sum(large_gaps)
        result.add_warning(f"Found {n_gaps} large gaps in wavelength coverage")
    
    return result


# =============================================================================
# Normalization Functions
# =============================================================================

def normalize_spectrum(
    flux: np.ndarray,
    error: Optional[np.ndarray],
    method: NormalizationMethod,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Normalize flux values using specified method.
    
    Args:
        flux: Flux array
        error: Error array (optional)
        method: Normalization method
        **kwargs: Method-specific parameters
        
    Returns:
        Tuple of (normalized_flux, normalized_error, normalization_params)
    """
    params = {}
    
    if method == NormalizationMethod.NONE:
        return flux.copy(), error.copy() if error is not None else None, params
    
    # Handle NaN values
    valid = np.isfinite(flux)
    if not np.any(valid):
        raise ValueError("No valid flux values for normalization")
    
    if method == NormalizationMethod.MIN_MAX:
        # Scale to [0, 1]
        vmin = np.nanmin(flux[valid])
        vmax = np.nanmax(flux[valid])
        
        if vmax == vmin:
            normalized = np.zeros_like(flux)
        else:
            normalized = (flux - vmin) / (vmax - vmin)
        
        params = {"min": float(vmin), "max": float(vmax)}
        
        if error is not None:
            normalized_error = error / (vmax - vmin)
        else:
            normalized_error = None
    
    elif method == NormalizationMethod.ZSCORE:
        # Standardize to mean=0, std=1
        mean = np.nanmean(flux[valid])
        std = np.nanstd(flux[valid])
        
        if std == 0:
            normalized = flux - mean
        else:
            normalized = (flux - mean) / std
        
        params = {"mean": float(mean), "std": float(std)}
        
        if error is not None:
            normalized_error = error / std if std > 0 else error
        else:
            normalized_error = None
    
    elif method == NormalizationMethod.MEDIAN:
        # Divide by median
        median = np.nanmedian(flux[valid])
        
        if median == 0:
            normalized = flux.copy()
        else:
            normalized = flux / median
        
        params = {"median": float(median)}
        
        if error is not None:
            normalized_error = error / median if median != 0 else error
        else:
            normalized_error = None
    
    elif method == NormalizationMethod.CONTINUUM:
        # Fit polynomial continuum and divide
        degree = kwargs.get("polynomial_degree", 3)
        
        # Use only valid points for fitting
        x = np.arange(len(flux))[valid]
        y = flux[valid]
        
        # Iterative sigma clipping for continuum fitting
        for _ in range(3):
            coeffs = np.polyfit(x, y, degree)
            continuum_fit = np.polyval(coeffs, x)
            residuals = y - continuum_fit
            sigma = np.std(residuals)
            mask = np.abs(residuals) < 3 * sigma
            x, y = x[mask], y[mask]
            if len(x) < degree + 1:
                break
        
        # Apply to full array
        full_x = np.arange(len(flux))
        continuum = np.polyval(coeffs, full_x)
        
        normalized = np.where(continuum != 0, flux / continuum, flux)
        
        params = {
            "continuum_coeffs": coeffs.tolist(),
            "polynomial_degree": degree,
        }
        
        if error is not None:
            normalized_error = np.where(continuum != 0, error / continuum, error)
        else:
            normalized_error = None
    
    elif method == NormalizationMethod.PERCENTILE:
        # Scale using percentiles (robust to outliers)
        p_low = kwargs.get("percentile_low", 5)
        p_high = kwargs.get("percentile_high", 95)
        
        vmin = np.nanpercentile(flux[valid], p_low)
        vmax = np.nanpercentile(flux[valid], p_high)
        
        if vmax == vmin:
            normalized = np.zeros_like(flux)
        else:
            normalized = (flux - vmin) / (vmax - vmin)
        
        params = {
            "percentile_low": p_low,
            "percentile_high": p_high,
            "value_low": float(vmin),
            "value_high": float(vmax),
        }
        
        if error is not None:
            normalized_error = error / (vmax - vmin) if vmax != vmin else error
        else:
            normalized_error = None
    
    elif method == NormalizationMethod.ROBUST:
        # Robust scaling using median and IQR
        median = np.nanmedian(flux[valid])
        q1 = np.nanpercentile(flux[valid], 25)
        q3 = np.nanpercentile(flux[valid], 75)
        iqr = q3 - q1
        
        if iqr == 0:
            normalized = flux - median
        else:
            normalized = (flux - median) / iqr
        
        params = {"median": float(median), "iqr": float(iqr), "q1": float(q1), "q3": float(q3)}
        
        if error is not None:
            normalized_error = error / iqr if iqr > 0 else error
        else:
            normalized_error = None
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, normalized_error, params


# =============================================================================
# Format Detection and Parsing
# =============================================================================

def detect_file_format(filepath: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Detect file format and extract format-specific info.
    
    Returns:
        Tuple of (format_type, format_info)
    """
    suffix = filepath.suffix.lower()
    
    if suffix in [".fits", ".fit"]:
        return "fits", {}
    
    elif suffix == ".json":
        return "json", {}
    
    elif suffix in [".csv", ".txt", ".dat", ".tbl"]:
        # Try to detect CSV dialect and structure
        with open(filepath, "r") as f:
            # Read first few lines
            lines = [f.readline() for _ in range(20)]
        
        # Detect delimiter
        sample = "".join(lines)
        if "\t" in sample:
            delimiter = "\t"
        elif "," in sample:
            delimiter = ","
        elif ";" in sample:
            delimiter = ";"
        else:
            delimiter = r"\s+"  # Whitespace
        
        # Detect header
        header_row = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                header_row = i + 1
            elif not line.strip():
                continue
            else:
                # Check if first data-like line
                try:
                    float(line.split()[0].replace(",", ""))
                    break
                except (ValueError, IndexError):
                    header_row = i + 1
        
        return "csv", {"delimiter": delimiter, "header_row": header_row}
    
    else:
        return "unknown", {}


def detect_instrument_from_data(
    wavelength: np.ndarray,
    metadata: Dict[str, Any],
) -> Instrument:
    """Detect instrument from wavelength range and metadata."""
    wl_min, wl_max = wavelength.min(), wavelength.max()
    
    # Check metadata for hints
    meta_str = json.dumps(metadata).lower()
    
    if "nirspec" in meta_str and "jwst" in meta_str:
        return Instrument.JWST_NIRSPEC
    elif "miri" in meta_str:
        return Instrument.JWST_MIRI
    elif "nircam" in meta_str:
        return Instrument.JWST_NIRCAM
    elif "niriss" in meta_str:
        return Instrument.JWST_NIRISS
    elif "wfc3" in meta_str:
        return Instrument.HST_WFC3
    elif "stis" in meta_str:
        return Instrument.HST_STIS
    elif "cos" in meta_str and "hst" in meta_str:
        return Instrument.HST_COS
    
    # Infer from wavelength range
    for instrument, (inst_min, inst_max) in INSTRUMENT_WAVELENGTH_RANGES.items():
        if instrument in [Instrument.GENERIC, Instrument.UNKNOWN]:
            continue
        
        # Check for significant overlap
        overlap_min = max(wl_min, inst_min)
        overlap_max = min(wl_max, inst_max)
        
        if overlap_max > overlap_min:
            overlap_fraction = (overlap_max - overlap_min) / (wl_max - wl_min)
            if overlap_fraction > 0.7:
                return instrument
    
    return Instrument.UNKNOWN


def find_column(df: pd.DataFrame, column_type: str) -> Optional[str]:
    """Find column matching a given type (wavelength, flux, error, etc.)."""
    candidates = COLUMN_MAPPINGS.get(column_type, [])
    
    for candidate in candidates:
        # Exact match
        if candidate in df.columns:
            return candidate
        
        # Case-insensitive match
        for col in df.columns:
            if col.lower() == candidate.lower():
                return col
    
    # Fuzzy match
    for candidate in candidates:
        for col in df.columns:
            if candidate.lower() in col.lower():
                return col
    
    return None


# =============================================================================
# Main Data Loader Class
# =============================================================================

class ExoplanetDataLoader:
    """
    Main data loader for exoplanet spectral datasets.
    
    Supports loading from various formats and instruments with automatic
    format detection, wavelength validation, and intensity normalization.
    
    Example usage:
        loader = ExoplanetDataLoader()
        spectrum = loader.load("path/to/spectrum.csv")
        
        # With specific options
        spectrum = loader.load(
            "path/to/spectrum.csv",
            instrument=Instrument.JWST_NIRSPEC,
            normalize=NormalizationMethod.MEDIAN,
            wavelength_unit=WavelengthUnit.MICRON,
        )
    """
    
    def __init__(
        self,
        default_instrument: Instrument = Instrument.GENERIC,
        default_normalization: NormalizationMethod = NormalizationMethod.NONE,
        strict_validation: bool = False,
    ):
        """
        Initialize the data loader.
        
        Args:
            default_instrument: Default instrument for validation
            default_normalization: Default normalization method
            strict_validation: Enable strict wavelength validation
        """
        self.default_instrument = default_instrument
        self.default_normalization = default_normalization
        self.strict_validation = strict_validation
    
    def load(
        self,
        filepath: Union[str, Path],
        instrument: Optional[Instrument] = None,
        normalize: Optional[NormalizationMethod] = None,
        wavelength_unit: Optional[WavelengthUnit] = None,
        flux_unit: Optional[FluxUnit] = None,
        wavelength_column: Optional[str] = None,
        flux_column: Optional[str] = None,
        error_column: Optional[str] = None,
        validate: bool = True,
        **kwargs,
    ) -> LoadedSpectrum:
        """
        Load a spectrum from file.
        
        Args:
            filepath: Path to spectrum file
            instrument: Instrument type (auto-detected if None)
            normalize: Normalization method (uses default if None)
            wavelength_unit: Wavelength unit (auto-detected if None)
            flux_unit: Flux unit
            wavelength_column: Override wavelength column name
            flux_column: Override flux column name
            error_column: Override error column name
            validate: Whether to validate wavelength range
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            LoadedSpectrum object with processed data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Spectrum file not found: {filepath}")
        
        # Detect format
        format_type, format_info = detect_file_format(filepath)
        
        # Load based on format
        if format_type == "fits":
            df, raw_metadata = self._load_fits(filepath, **kwargs)
        elif format_type == "json":
            df, raw_metadata = self._load_json(filepath, **kwargs)
        elif format_type == "csv":
            df, raw_metadata = self._load_csv(filepath, format_info, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")
        
        # Find columns
        wl_col = wavelength_column or find_column(df, "wavelength")
        flux_col = flux_column or find_column(df, "flux")
        err_col = error_column or find_column(df, "error")
        qual_col = find_column(df, "quality")
        
        if wl_col is None:
            raise ValueError("Could not identify wavelength column")
        if flux_col is None:
            raise ValueError("Could not identify flux column")
        
        # Extract arrays
        wavelength = df[wl_col].values.astype(np.float64)
        flux = df[flux_col].values.astype(np.float64)
        error = df[err_col].values.astype(np.float64) if err_col else None
        quality = df[qual_col].values if qual_col else None
        
        # Detect and convert wavelength unit
        if wavelength_unit is None:
            wavelength_unit = detect_wavelength_unit(wavelength)
        
        wavelength = convert_wavelength_to_microns(wavelength, wavelength_unit)
        
        # Detect instrument
        if instrument is None:
            instrument = detect_instrument_from_data(wavelength, raw_metadata)
        
        # Validate wavelength range
        if validate:
            validation = validate_wavelength_range(
                wavelength, instrument, strict=self.strict_validation
            )
            
            for error_msg in validation.errors:
                logger.error(error_msg)
            for warning_msg in validation.warnings:
                logger.warning(warning_msg)
            
            if not validation.is_valid and self.strict_validation:
                raise ValueError(
                    f"Wavelength validation failed: {'; '.join(validation.errors)}"
                )
        
        # Sort by wavelength
        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        flux = flux[sort_idx]
        if error is not None:
            error = error[sort_idx]
        if quality is not None:
            quality = quality[sort_idx]
        
        # Store original before normalization
        original_flux = flux.copy()
        original_error = error.copy() if error is not None else None
        
        # Normalize
        norm_method = normalize if normalize is not None else self.default_normalization
        flux, error, norm_params = normalize_spectrum(
            flux, error, norm_method, **kwargs
        )
        
        # Create metadata
        metadata = SpectrumMetadata(
            source_file=str(filepath),
            instrument=instrument,
            wavelength_unit=WavelengthUnit.MICRON,  # Always microns after conversion
            flux_unit=flux_unit or FluxUnit.PPM,
            wavelength_range=(float(wavelength.min()), float(wavelength.max())),
            n_points=len(wavelength),
            additional_info=raw_metadata,
        )
        
        # Extract target name from metadata if available
        for key in ["target", "object", "target_name", "OBJECT", "TARGET"]:
            if key in raw_metadata:
                metadata.target_name = str(raw_metadata[key])
                break
        
        return LoadedSpectrum(
            wavelength=wavelength,
            flux=flux,
            error=error,
            quality_flags=quality,
            metadata=metadata,
            original_flux=original_flux,
            original_error=original_error,
            normalization_method=norm_method,
            normalization_params=norm_params,
        )
    
    def _load_csv(
        self,
        filepath: Path,
        format_info: Dict[str, Any],
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load CSV/text file."""
        delimiter = format_info.get("delimiter", ",")
        header_row = format_info.get("header_row", 0)
        
        # Handle comment lines
        comment = kwargs.get("comment", "#")
        
        try:
            if delimiter == r"\s+":
                df = pd.read_csv(
                    filepath,
                    delim_whitespace=True,
                    header=header_row,
                    comment=comment,
                )
            else:
                df = pd.read_csv(
                    filepath,
                    delimiter=delimiter,
                    header=header_row,
                    comment=comment,
                )
        except Exception as e:
            # Fallback: try without header
            df = pd.read_csv(filepath, header=None, comment=comment)
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
        
        # Extract metadata from comments
        metadata = {}
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("#"):
                    # Try to parse key=value pairs
                    match = re.match(r"#\s*(\w+)\s*[=:]\s*(.+)", line)
                    if match:
                        metadata[match.group(1)] = match.group(2).strip()
                elif line.strip():
                    break
        
        return df, metadata
    
    def _load_json(
        self,
        filepath: Path,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of records
            df = pd.DataFrame(data)
            metadata = {}
        elif isinstance(data, dict):
            # Check for nested structure
            if "data" in data:
                df = pd.DataFrame(data["data"])
                metadata = {k: v for k, v in data.items() if k != "data"}
            elif "wavelength" in data or "wave" in data:
                # Column-oriented
                df = pd.DataFrame(data)
                metadata = {}
            else:
                # Try to extract arrays
                df = pd.DataFrame(data)
                metadata = {}
        else:
            raise ValueError("Unsupported JSON structure")
        
        return df, metadata
    
    def _load_fits(
        self,
        filepath: Path,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load FITS file."""
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("astropy is required for FITS file support")
        
        hdu_index = kwargs.get("hdu_index", 1)
        
        with fits.open(filepath) as hdul:
            # Extract header metadata
            header = hdul[0].header
            metadata = {k: v for k, v in header.items() if k and v}
            
            # Try to find data HDU
            if len(hdul) > hdu_index:
                data_hdu = hdul[hdu_index]
            else:
                data_hdu = hdul[0]
            
            if hasattr(data_hdu.data, 'names'):
                # Table data
                df = pd.DataFrame({
                    name: data_hdu.data[name]
                    for name in data_hdu.data.names
                })
            else:
                # Image data - assume rows are different quantities
                data = data_hdu.data
                if data.ndim == 1:
                    df = pd.DataFrame({"flux": data})
                elif data.ndim == 2:
                    if data.shape[0] <= 10:
                        # Few rows = different quantities
                        columns = ["wavelength", "flux", "error"][:data.shape[0]]
                        df = pd.DataFrame({
                            col: data[i] for i, col in enumerate(columns)
                        })
                    else:
                        # Many rows = spectral axis
                        df = pd.DataFrame({"flux": data[:, 0] if data.ndim > 1 else data})
                else:
                    raise ValueError(f"Unsupported FITS data shape: {data.shape}")
        
        return df, metadata
    
    def load_multiple(
        self,
        filepaths: List[Union[str, Path]],
        **kwargs,
    ) -> List[LoadedSpectrum]:
        """
        Load multiple spectra.
        
        Args:
            filepaths: List of file paths
            **kwargs: Arguments passed to load()
            
        Returns:
            List of LoadedSpectrum objects
        """
        spectra = []
        
        for filepath in filepaths:
            try:
                spectrum = self.load(filepath, **kwargs)
                spectra.append(spectrum)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        
        return spectra
    
    def load_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.csv",
        **kwargs,
    ) -> List[LoadedSpectrum]:
        """
        Load all spectra from a directory matching a pattern.
        
        Args:
            directory: Directory path
            pattern: Glob pattern for file matching
            **kwargs: Arguments passed to load()
            
        Returns:
            List of LoadedSpectrum objects
        """
        directory = Path(directory)
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        filepaths = sorted(directory.glob(pattern))
        
        return self.load_multiple(filepaths, **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_spectrum(
    filepath: Union[str, Path],
    **kwargs,
) -> LoadedSpectrum:
    """
    Convenience function to load a single spectrum.
    
    Args:
        filepath: Path to spectrum file
        **kwargs: Arguments passed to ExoplanetDataLoader.load()
        
    Returns:
        LoadedSpectrum object
    """
    loader = ExoplanetDataLoader()
    return loader.load(filepath, **kwargs)


def load_jwst_spectrum(
    filepath: Union[str, Path],
    instrument: Instrument = Instrument.JWST_NIRSPEC,
    **kwargs,
) -> LoadedSpectrum:
    """Load a JWST spectrum with appropriate defaults."""
    loader = ExoplanetDataLoader(
        default_instrument=instrument,
        strict_validation=True,
    )
    return loader.load(filepath, instrument=instrument, **kwargs)


def load_hubble_spectrum(
    filepath: Union[str, Path],
    instrument: Instrument = Instrument.HST_WFC3,
    **kwargs,
) -> LoadedSpectrum:
    """Load a Hubble spectrum with appropriate defaults."""
    loader = ExoplanetDataLoader(
        default_instrument=instrument,
        strict_validation=True,
    )
    return loader.load(filepath, instrument=instrument, **kwargs)


# =============================================================================
# Main (Example Usage)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <spectrum_file>")
        print("\nExample:")
        print("  python data_loader.py spectrum.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load spectrum
    loader = ExoplanetDataLoader()
    
    try:
        spectrum = loader.load(
            filepath,
            normalize=NormalizationMethod.MEDIAN,
            validate=True,
        )
        
        print(f"\n{'='*60}")
        print(f"Loaded: {filepath}")
        print(f"{'='*60}")
        print(f"Instrument: {spectrum.metadata.instrument.value}")
        print(f"Wavelength range: {spectrum.wavelength_range[0]:.3f} - {spectrum.wavelength_range[1]:.3f} μm")
        print(f"Data points: {len(spectrum.wavelength)}")
        print(f"SNR: {spectrum.snr:.1f}" if spectrum.snr else "SNR: N/A")
        print(f"Normalization: {spectrum.normalization_method.value}")
        
        if spectrum.metadata.target_name:
            print(f"Target: {spectrum.metadata.target_name}")
        
        # Show first few data points
        print(f"\nFirst 5 data points:")
        df = spectrum.to_dataframe().head()
        print(df.to_string(index=False))
        
    except Exception as e:
        print(f"Error loading spectrum: {e}")
        sys.exit(1)
