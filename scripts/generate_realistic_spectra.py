#!/usr/bin/env python
"""
Generate Realistic Synthetic Exoplanet Transmission Spectra

This script generates physically-motivated synthetic transmission spectra
including:
- Stellar baseline (blackbody + limb darkening)
- Planetary atmospheric absorption features
- Molecular bands: H2O, CO2, CH4, Na, O2, and more
- Realistic noise models (photon noise, detector noise, systematics)
- Visible + infrared wavelength coverage (0.3 - 15 μm)

Output: CSV files with wavelength, intensity, error, and metadata
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Physical Constants
# =============================================================================

PLANCK_H = 6.62607015e-34      # Planck constant (J·s)
SPEED_OF_LIGHT = 2.99792458e8  # Speed of light (m/s)
BOLTZMANN_K = 1.380649e-23     # Boltzmann constant (J/K)
STEFAN_BOLTZMANN = 5.670374e-8 # Stefan-Boltzmann constant (W/m²/K⁴)

# Solar/Earth reference values
R_SUN = 6.957e8       # Solar radius (m)
R_EARTH = 6.371e6     # Earth radius (m)
R_JUPITER = 6.9911e7  # Jupiter radius (m)


# =============================================================================
# Molecular Absorption Data
# =============================================================================

@dataclass
class MolecularBand:
    """Represents a molecular absorption band."""
    center: float          # Band center wavelength (μm)
    width: float           # Band width (μm)
    strength: float        # Relative absorption strength (0-1)
    shape: str = "gaussian"  # Band shape: gaussian, lorentzian, voigt


# Comprehensive molecular absorption features
# Based on HITRAN database and spectroscopic studies
MOLECULAR_FEATURES: Dict[str, List[MolecularBand]] = {
    "H2O": [
        # Water vapor - dominant in many exoplanet atmospheres
        MolecularBand(0.72, 0.02, 0.15),    # Visible band
        MolecularBand(0.82, 0.03, 0.20),    # Near-IR
        MolecularBand(0.94, 0.04, 0.35),    # Strong NIR band
        MolecularBand(1.14, 0.06, 0.45),    # J-band feature
        MolecularBand(1.38, 0.08, 0.70),    # Strong H-band
        MolecularBand(1.87, 0.12, 0.80),    # K-band
        MolecularBand(2.70, 0.25, 0.95),    # Fundamental stretch
        MolecularBand(5.50, 0.40, 0.60),    # Mid-IR bending
        MolecularBand(6.27, 0.35, 0.85),    # Strong mid-IR
    ],
    "CO2": [
        # Carbon dioxide - key biosignature/climate indicator
        MolecularBand(1.43, 0.03, 0.25),    # Weak overtone
        MolecularBand(1.60, 0.04, 0.30),    # Overtone
        MolecularBand(2.00, 0.06, 0.40),    # Combination band
        MolecularBand(2.70, 0.10, 0.55),    # Asymmetric stretch
        MolecularBand(4.26, 0.20, 0.95),    # Fundamental (strongest)
        MolecularBand(9.40, 0.30, 0.35),    # Hot band
        MolecularBand(10.40, 0.25, 0.40),   # Isotope band
        MolecularBand(15.00, 0.80, 0.75),   # Bending mode
    ],
    "CH4": [
        # Methane - key for hot Jupiters and biosignatures
        MolecularBand(0.89, 0.02, 0.20),    # Visible band
        MolecularBand(1.00, 0.03, 0.25),    # Near-IR
        MolecularBand(1.16, 0.04, 0.30),    # J-band
        MolecularBand(1.38, 0.05, 0.35),    # Overlap with H2O
        MolecularBand(1.66, 0.06, 0.50),    # H-band feature
        MolecularBand(2.20, 0.08, 0.55),    # K-band
        MolecularBand(2.32, 0.07, 0.60),    # Strong K-band
        MolecularBand(3.30, 0.15, 0.90),    # C-H stretch (strongest)
        MolecularBand(7.66, 0.40, 0.70),    # Mid-IR bending
    ],
    "Na": [
        # Sodium - prominent in hot Jupiter atmospheres
        MolecularBand(0.5890, 0.002, 0.85, "lorentzian"),  # D2 line
        MolecularBand(0.5896, 0.002, 0.80, "lorentzian"),  # D1 line
        # Pressure-broadened wings
        MolecularBand(0.5893, 0.015, 0.40),  # Broad wing
    ],
    "K": [
        # Potassium - often seen with sodium
        MolecularBand(0.7665, 0.002, 0.70, "lorentzian"),  # D1 line
        MolecularBand(0.7699, 0.002, 0.65, "lorentzian"),  # D2 line
        MolecularBand(0.7682, 0.012, 0.30),  # Pressure wings
    ],
    "O2": [
        # Oxygen - biosignature for Earth-like planets
        MolecularBand(0.6275, 0.003, 0.25),  # B-band
        MolecularBand(0.6867, 0.004, 0.35),  # B-band
        MolecularBand(0.7620, 0.006, 0.65),  # A-band (strongest visible)
        MolecularBand(1.27, 0.02, 0.45),     # NIR band
    ],
    "O3": [
        # Ozone - UV/visible absorber, biosignature
        MolecularBand(0.32, 0.03, 0.80),     # Hartley band (UV)
        MolecularBand(0.55, 0.08, 0.25),     # Chappuis band (visible)
        MolecularBand(0.60, 0.10, 0.30),     # Chappuis band
        MolecularBand(9.60, 0.50, 0.85),     # Mid-IR (strongest)
    ],
    "CO": [
        # Carbon monoxide - hot Jupiter indicator
        MolecularBand(2.35, 0.08, 0.55),     # First overtone
        MolecularBand(4.67, 0.15, 0.85),     # Fundamental (strongest)
    ],
    "NH3": [
        # Ammonia - cool gas giant atmospheres
        MolecularBand(1.50, 0.05, 0.30),     # Overtone
        MolecularBand(2.00, 0.08, 0.45),     # Combination
        MolecularBand(2.25, 0.06, 0.40),     # NIR
        MolecularBand(3.00, 0.12, 0.60),     # N-H stretch
        MolecularBand(6.10, 0.30, 0.50),     # Umbrella mode
        MolecularBand(10.35, 0.40, 0.75),    # Strongest mid-IR
    ],
    "TiO": [
        # Titanium oxide - ultra-hot Jupiter marker
        MolecularBand(0.52, 0.02, 0.45),     # Visible bands
        MolecularBand(0.57, 0.03, 0.55),
        MolecularBand(0.62, 0.03, 0.60),
        MolecularBand(0.67, 0.04, 0.65),
        MolecularBand(0.71, 0.04, 0.50),
    ],
    "VO": [
        # Vanadium oxide - ultra-hot Jupiter
        MolecularBand(0.55, 0.02, 0.35),
        MolecularBand(0.74, 0.03, 0.45),
        MolecularBand(0.79, 0.03, 0.50),
        MolecularBand(0.85, 0.04, 0.40),
    ],
    "FeH": [
        # Iron hydride - L/T dwarf and brown dwarf atmospheres
        MolecularBand(0.87, 0.02, 0.30),
        MolecularBand(0.99, 0.03, 0.45),
        MolecularBand(1.20, 0.04, 0.40),
        MolecularBand(1.60, 0.05, 0.35),
    ],
}


# =============================================================================
# Stellar and Planetary Parameters
# =============================================================================

@dataclass
class StellarParams:
    """Stellar parameters for the host star."""
    temperature: float = 5800.0      # Effective temperature (K)
    radius: float = 1.0              # Radius in solar radii
    mass: float = 1.0                # Mass in solar masses
    metallicity: float = 0.0         # [Fe/H]
    spectral_type: str = "G2V"       # Spectral classification
    limb_darkening_coeffs: Tuple[float, float] = (0.4, 0.26)  # Quadratic LD


@dataclass
class PlanetaryParams:
    """Planetary and atmospheric parameters."""
    name: str = "Synthetic-1b"
    radius: float = 1.0              # Radius in Jupiter radii
    mass: float = 1.0                # Mass in Jupiter masses
    orbital_period: float = 3.5      # Orbital period (days)
    semi_major_axis: float = 0.05    # Semi-major axis (AU)
    equilibrium_temp: float = 1200.0 # Equilibrium temperature (K)
    
    # Atmospheric composition (log10 mixing ratios)
    h2o_abundance: float = -3.0
    co2_abundance: float = -4.0
    ch4_abundance: float = -5.0
    co_abundance: float = -4.5
    na_abundance: float = -6.0
    k_abundance: float = -6.5
    o2_abundance: float = -6.0
    o3_abundance: float = -7.0
    nh3_abundance: float = -6.0
    tio_abundance: float = -8.0
    vo_abundance: float = -8.0
    feh_abundance: float = -7.0
    
    # Cloud properties
    cloud_top_pressure: float = 0.01  # Cloud top pressure (bar)
    cloud_fraction: float = 0.3       # Cloud coverage fraction
    
    # Scale height
    mean_molecular_weight: float = 2.3  # Mean molecular weight (amu)


@dataclass
class ObservationParams:
    """Observational parameters affecting the spectrum."""
    wavelength_min: float = 0.3       # Minimum wavelength (μm)
    wavelength_max: float = 15.0      # Maximum wavelength (μm)
    resolution: int = 2000            # Spectral resolution points
    snr: float = 50.0                 # Signal-to-noise ratio
    
    # Noise components
    photon_noise: bool = True
    detector_noise: float = 5.0       # Detector noise (ppm)
    systematic_amplitude: float = 20.0 # Systematic noise amplitude (ppm)
    
    # Instrumental effects
    spectral_resolution: float = 100.0  # R = λ/Δλ
    apply_instrumental_broadening: bool = True


# =============================================================================
# Spectrum Generation Functions
# =============================================================================

def planck_function(wavelength_um: np.ndarray, temperature: float) -> np.ndarray:
    """
    Calculate Planck blackbody spectrum.
    
    Args:
        wavelength_um: Wavelength in microns
        temperature: Temperature in Kelvin
        
    Returns:
        Spectral radiance (arbitrary units, normalized)
    """
    wavelength_m = wavelength_um * 1e-6
    
    # Planck function
    c1 = 2 * PLANCK_H * SPEED_OF_LIGHT**2
    c2 = PLANCK_H * SPEED_OF_LIGHT / BOLTZMANN_K
    
    # Avoid overflow in exponential
    exponent = np.clip(c2 / (wavelength_m * temperature), 0, 700)
    
    radiance = c1 / (wavelength_m**5 * (np.exp(exponent) - 1))
    
    # Normalize
    return radiance / radiance.max()


def apply_limb_darkening(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    coeffs: Tuple[float, float],
    mu: float = 0.5,
) -> np.ndarray:
    """
    Apply quadratic limb darkening to stellar spectrum.
    
    Uses the quadratic law: I(μ)/I(1) = 1 - u1*(1-μ) - u2*(1-μ)²
    
    Args:
        spectrum: Input spectrum
        wavelength: Wavelength array
        coeffs: Limb darkening coefficients (u1, u2)
        mu: Cosine of angle from disk center (0=limb, 1=center)
        
    Returns:
        Limb-darkened spectrum
    """
    u1, u2 = coeffs
    
    # Wavelength-dependent limb darkening (stronger in blue)
    wl_factor = 1.0 + 0.3 * (1.0 - wavelength / wavelength.max())
    
    ld_factor = 1 - u1 * wl_factor * (1 - mu) - u2 * wl_factor * (1 - mu)**2
    
    return spectrum * ld_factor


def gaussian_absorption(
    wavelength: np.ndarray,
    center: float,
    width: float,
    depth: float,
) -> np.ndarray:
    """Generate Gaussian absorption profile."""
    return depth * np.exp(-((wavelength - center)**2) / (2 * width**2))


def lorentzian_absorption(
    wavelength: np.ndarray,
    center: float,
    width: float,
    depth: float,
) -> np.ndarray:
    """Generate Lorentzian absorption profile (pressure-broadened lines)."""
    gamma = width / 2
    return depth * (gamma**2) / ((wavelength - center)**2 + gamma**2)


def voigt_absorption(
    wavelength: np.ndarray,
    center: float,
    width_g: float,
    width_l: float,
    depth: float,
) -> np.ndarray:
    """
    Generate Voigt profile (convolution of Gaussian and Lorentzian).
    Approximation using pseudo-Voigt profile.
    """
    # Pseudo-Voigt approximation
    f_g = 2.355 * width_g  # FWHM for Gaussian
    f_l = 2 * width_l      # FWHM for Lorentzian
    
    f_v = (f_g**5 + 2.69*f_g**4*f_l + 2.43*f_g**3*f_l**2 + 
           4.47*f_g**2*f_l**3 + 0.078*f_g*f_l**4 + f_l**5) ** 0.2
    
    eta = 1.37 * (f_l/f_v) - 0.477 * (f_l/f_v)**2 + 0.11 * (f_l/f_v)**3
    
    gauss = gaussian_absorption(wavelength, center, width_g, 1.0)
    lorentz = lorentzian_absorption(wavelength, center, width_l, 1.0)
    
    return depth * (eta * lorentz + (1 - eta) * gauss)


def calculate_scale_height(
    temperature: float,
    gravity: float,
    mu: float,
) -> float:
    """
    Calculate atmospheric scale height.
    
    H = kT / (μ * m_H * g)
    
    Args:
        temperature: Atmospheric temperature (K)
        gravity: Surface gravity (m/s²)
        mu: Mean molecular weight (amu)
        
    Returns:
        Scale height in meters
    """
    m_H = 1.6735e-27  # Hydrogen atom mass (kg)
    return BOLTZMANN_K * temperature / (mu * m_H * gravity)


def generate_molecular_absorption(
    wavelength: np.ndarray,
    molecule: str,
    abundance: float,
    temperature: float,
    pressure_broadening: float = 1.0,
) -> np.ndarray:
    """
    Generate absorption spectrum for a given molecule.
    
    Args:
        wavelength: Wavelength array (μm)
        molecule: Molecule name (e.g., "H2O", "CO2")
        abundance: Log10 mixing ratio
        temperature: Atmospheric temperature (K)
        pressure_broadening: Pressure broadening factor
        
    Returns:
        Absorption spectrum (0 = no absorption, 1 = full absorption)
    """
    absorption = np.zeros_like(wavelength)
    
    if molecule not in MOLECULAR_FEATURES:
        return absorption
    
    # Temperature scaling for line widths
    temp_factor = np.sqrt(temperature / 1000.0)
    
    # Abundance scaling (exponential of log abundance)
    abundance_factor = 10 ** (abundance + 4)  # Normalize to typical values
    abundance_factor = np.clip(abundance_factor, 0.001, 100)
    
    for band in MOLECULAR_FEATURES[molecule]:
        # Check if band is within wavelength range
        if not (wavelength.min() <= band.center <= wavelength.max()):
            continue
        
        # Scale width with temperature and pressure
        effective_width = band.width * temp_factor * pressure_broadening
        
        # Scale strength with abundance
        effective_strength = band.strength * abundance_factor
        effective_strength = np.clip(effective_strength, 0, 1)
        
        # Generate appropriate line shape
        if band.shape == "gaussian":
            absorption += gaussian_absorption(
                wavelength, band.center, effective_width, effective_strength
            )
        elif band.shape == "lorentzian":
            absorption += lorentzian_absorption(
                wavelength, band.center, effective_width, effective_strength
            )
        elif band.shape == "voigt":
            absorption += voigt_absorption(
                wavelength, band.center, effective_width * 0.7,
                effective_width * 0.3, effective_strength
            )
    
    return np.clip(absorption, 0, 1)


def generate_cloud_opacity(
    wavelength: np.ndarray,
    cloud_fraction: float,
    cloud_top: float,
) -> np.ndarray:
    """
    Generate wavelength-dependent cloud opacity.
    
    Clouds typically have gray opacity with slight wavelength dependence.
    """
    # Slight wavelength dependence (Rayleigh-like at short wavelengths)
    rayleigh_scattering = 0.1 * (0.5 / wavelength) ** 4
    rayleigh_scattering = np.clip(rayleigh_scattering, 0, 0.5)
    
    # Gray cloud opacity
    gray_opacity = cloud_fraction * 0.3
    
    return gray_opacity + rayleigh_scattering * cloud_fraction


def generate_stellar_spectrum(
    wavelength: np.ndarray,
    stellar: StellarParams,
) -> np.ndarray:
    """
    Generate stellar baseline spectrum.
    
    Args:
        wavelength: Wavelength array (μm)
        stellar: Stellar parameters
        
    Returns:
        Normalized stellar spectrum
    """
    # Blackbody baseline
    spectrum = planck_function(wavelength, stellar.temperature)
    
    # Apply limb darkening
    spectrum = apply_limb_darkening(
        spectrum, wavelength, stellar.limb_darkening_coeffs
    )
    
    # Add stellar absorption lines (simplified)
    # These would be more detailed in a real model
    if stellar.temperature < 4500:
        # Cool star - TiO/VO bands
        spectrum *= (1 - 0.1 * generate_molecular_absorption(
            wavelength, "TiO", -6, stellar.temperature
        ))
    
    return spectrum


def generate_transmission_spectrum(
    wavelength: np.ndarray,
    stellar: StellarParams,
    planet: PlanetaryParams,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate full transmission spectrum.
    
    Args:
        wavelength: Wavelength array (μm)
        stellar: Stellar parameters
        planet: Planetary parameters
        
    Returns:
        Tuple of (transmission spectrum, component spectra dict)
    """
    # Start with stellar spectrum
    stellar_spectrum = generate_stellar_spectrum(wavelength, stellar)
    
    # Calculate transit depth baseline (Rp/Rs)²
    rp_meters = planet.radius * R_JUPITER
    rs_meters = stellar.radius * R_SUN
    transit_depth_baseline = (rp_meters / rs_meters) ** 2
    
    # Calculate atmospheric contribution
    # Effective atmosphere height adds to transit depth
    gravity = 24.79 * planet.mass / planet.radius**2  # Surface gravity (m/s²)
    scale_height = calculate_scale_height(
        planet.equilibrium_temp, gravity, planet.mean_molecular_weight
    )
    
    # Number of scale heights contributing to absorption
    n_scale_heights = 5
    atmosphere_height = n_scale_heights * scale_height
    
    # Relative atmosphere contribution to transit depth
    delta_depth = 2 * rp_meters * atmosphere_height / rs_meters**2
    
    # Generate molecular absorption components
    components = {}
    total_absorption = np.zeros_like(wavelength)
    
    molecules_and_abundances = [
        ("H2O", planet.h2o_abundance),
        ("CO2", planet.co2_abundance),
        ("CH4", planet.ch4_abundance),
        ("CO", planet.co_abundance),
        ("Na", planet.na_abundance),
        ("K", planet.k_abundance),
        ("O2", planet.o2_abundance),
        ("O3", planet.o3_abundance),
        ("NH3", planet.nh3_abundance),
        ("TiO", planet.tio_abundance),
        ("VO", planet.vo_abundance),
        ("FeH", planet.feh_abundance),
    ]
    
    for molecule, abundance in molecules_and_abundances:
        if abundance > -10:  # Only include if significant
            absorption = generate_molecular_absorption(
                wavelength, molecule, abundance, planet.equilibrium_temp
            )
            components[molecule] = absorption
            total_absorption += absorption
    
    # Normalize and clip total absorption
    total_absorption = np.clip(total_absorption, 0, 1)
    
    # Add cloud opacity
    cloud_opacity = generate_cloud_opacity(
        wavelength, planet.cloud_fraction, planet.cloud_top_pressure
    )
    components["clouds"] = cloud_opacity
    
    # Combined opacity (clouds reduce molecular feature depth)
    effective_absorption = total_absorption * (1 - cloud_opacity) + cloud_opacity * 0.5
    
    # Calculate transmission spectrum
    # Transit depth variation in ppm
    transit_depth_ppm = transit_depth_baseline * 1e6
    depth_variation_ppm = delta_depth * 1e6 * effective_absorption
    
    transmission_spectrum = transit_depth_ppm + depth_variation_ppm
    
    # Store components
    components["stellar"] = stellar_spectrum
    components["baseline_ppm"] = np.full_like(wavelength, transit_depth_ppm)
    
    return transmission_spectrum, components


def add_noise(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    obs: ObservationParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic noise to spectrum.
    
    Args:
        spectrum: Clean spectrum
        wavelength: Wavelength array
        obs: Observation parameters
        
    Returns:
        Tuple of (noisy spectrum, error array)
    """
    noise_components = []
    
    # 1. Photon noise (wavelength-dependent, lower SNR in IR)
    if obs.photon_noise:
        # SNR decreases toward longer wavelengths
        snr_wavelength = obs.snr * np.sqrt(
            planck_function(wavelength, 5800) / 
            planck_function(wavelength, 5800).max()
        )
        snr_wavelength = np.clip(snr_wavelength, obs.snr * 0.3, obs.snr)
        
        photon_noise = np.random.normal(0, spectrum.std() / snr_wavelength)
        noise_components.append(photon_noise)
    
    # 2. Detector noise (white noise)
    if obs.detector_noise > 0:
        detector = np.random.normal(0, obs.detector_noise, len(spectrum))
        noise_components.append(detector)
    
    # 3. Systematic noise (correlated, low-frequency)
    if obs.systematic_amplitude > 0:
        # Generate correlated noise using random walk
        random_walk = np.cumsum(np.random.normal(0, 1, len(spectrum)))
        systematic = random_walk - np.mean(random_walk)
        systematic = systematic / np.std(systematic) * obs.systematic_amplitude
        
        # Add polynomial trend
        x = np.linspace(-1, 1, len(spectrum))
        trend_coeffs = np.random.normal(0, obs.systematic_amplitude / 3, 3)
        trend = np.polyval(trend_coeffs, x)
        
        noise_components.append(systematic * 0.3 + trend * 0.7)
    
    # Combine noise
    total_noise = sum(noise_components)
    noisy_spectrum = spectrum + total_noise
    
    # Estimate error (standard deviation of noise)
    error = np.sqrt(
        (spectrum.std() / obs.snr) ** 2 + 
        obs.detector_noise ** 2 +
        (obs.systematic_amplitude * 0.1) ** 2
    )
    error_array = np.full_like(spectrum, error)
    
    # Add wavelength-dependent error component
    error_array *= (1 + 0.3 * (1 - planck_function(wavelength, 5800)))
    
    return noisy_spectrum, error_array


def apply_instrumental_broadening(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """
    Apply instrumental spectral broadening.
    
    Args:
        spectrum: Input spectrum
        wavelength: Wavelength array
        resolution: Spectral resolution R = λ/Δλ
        
    Returns:
        Broadened spectrum
    """
    # Calculate wavelength step
    dwavelength = np.median(np.diff(wavelength))
    mean_wavelength = np.mean(wavelength)
    
    # Resolution element in wavelength units
    resolution_element = mean_wavelength / resolution
    
    # Gaussian smoothing sigma in pixels
    sigma_pixels = resolution_element / dwavelength / 2.355
    
    if sigma_pixels > 0.5:
        return gaussian_filter1d(spectrum, sigma_pixels)
    return spectrum


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_spectrum(
    stellar: Optional[StellarParams] = None,
    planet: Optional[PlanetaryParams] = None,
    obs: Optional[ObservationParams] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate a complete synthetic transmission spectrum.
    
    Args:
        stellar: Stellar parameters (uses defaults if None)
        planet: Planetary parameters (uses defaults if None)
        obs: Observation parameters (uses defaults if None)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (DataFrame with spectrum, metadata dict)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use defaults if not provided
    stellar = stellar or StellarParams()
    planet = planet or PlanetaryParams()
    obs = obs or ObservationParams()
    
    # Generate wavelength grid
    wavelength = np.linspace(obs.wavelength_min, obs.wavelength_max, obs.resolution)
    
    # Generate transmission spectrum
    spectrum, components = generate_transmission_spectrum(wavelength, stellar, planet)
    
    # Apply instrumental broadening
    if obs.apply_instrumental_broadening:
        spectrum = apply_instrumental_broadening(
            spectrum, wavelength, obs.spectral_resolution
        )
    
    # Store clean spectrum
    clean_spectrum = spectrum.copy()
    
    # Add noise
    noisy_spectrum, error = add_noise(spectrum, wavelength, obs)
    
    # Create DataFrame
    df = pd.DataFrame({
        "wavelength_um": wavelength,
        "transit_depth_ppm": noisy_spectrum,
        "error_ppm": error,
        "clean_spectrum_ppm": clean_spectrum,
    })
    
    # Add molecular contributions
    for mol, absorption in components.items():
        if mol not in ["stellar", "baseline_ppm", "clouds"]:
            df[f"absorption_{mol}"] = absorption
    
    # Create metadata
    metadata = {
        "stellar": {
            "temperature": stellar.temperature,
            "radius": stellar.radius,
            "spectral_type": stellar.spectral_type,
        },
        "planetary": {
            "name": planet.name,
            "radius_rjup": planet.radius,
            "equilibrium_temp": planet.equilibrium_temp,
            "h2o_abundance": planet.h2o_abundance,
            "co2_abundance": planet.co2_abundance,
            "ch4_abundance": planet.ch4_abundance,
            "na_abundance": planet.na_abundance,
            "o2_abundance": planet.o2_abundance,
        },
        "observation": {
            "wavelength_range": [obs.wavelength_min, obs.wavelength_max],
            "resolution_points": obs.resolution,
            "snr": obs.snr,
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    return df, metadata


def save_spectrum(
    df: pd.DataFrame,
    metadata: Dict,
    output_path: Path,
    name: str = "synthetic_spectrum",
) -> Tuple[Path, Path]:
    """
    Save spectrum to CSV and metadata to JSON.
    
    Args:
        df: Spectrum DataFrame
        metadata: Metadata dictionary
        output_path: Output directory
        name: Base filename
        
    Returns:
        Tuple of (CSV path, JSON path)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / f"{name}.csv"
    json_path = output_path / f"{name}_metadata.json"
    
    # Save CSV
    df.to_csv(csv_path, index=False, float_format="%.6f")
    
    # Save metadata
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return csv_path, json_path


# =============================================================================
# Planet Presets
# =============================================================================

def create_hot_jupiter() -> Tuple[StellarParams, PlanetaryParams]:
    """Create parameters for a typical hot Jupiter system."""
    stellar = StellarParams(
        temperature=6000,
        radius=1.2,
        spectral_type="F8V",
    )
    planet = PlanetaryParams(
        name="Hot-Jupiter-1b",
        radius=1.4,
        mass=1.2,
        equilibrium_temp=1800,
        h2o_abundance=-3.5,
        co2_abundance=-5.0,
        ch4_abundance=-7.0,  # Destroyed at high temps
        co_abundance=-3.5,
        na_abundance=-5.0,
        k_abundance=-5.5,
        tio_abundance=-7.0,
        vo_abundance=-7.5,
        cloud_fraction=0.2,
    )
    return stellar, planet


def create_warm_neptune() -> Tuple[StellarParams, PlanetaryParams]:
    """Create parameters for a warm Neptune system."""
    stellar = StellarParams(
        temperature=5200,
        radius=0.85,
        spectral_type="K2V",
    )
    planet = PlanetaryParams(
        name="Warm-Neptune-1b",
        radius=0.35,  # ~4 Earth radii
        mass=0.05,    # ~15 Earth masses
        equilibrium_temp=800,
        h2o_abundance=-3.0,
        co2_abundance=-4.5,
        ch4_abundance=-4.0,
        co_abundance=-5.0,
        na_abundance=-6.5,
        cloud_fraction=0.5,
        mean_molecular_weight=4.0,
    )
    return stellar, planet


def create_earth_like() -> Tuple[StellarParams, PlanetaryParams]:
    """Create parameters for an Earth-like planet."""
    stellar = StellarParams(
        temperature=5800,
        radius=1.0,
        spectral_type="G2V",
    )
    planet = PlanetaryParams(
        name="Earth-Analog-1b",
        radius=0.09,   # ~1 Earth radius
        mass=0.003,    # ~1 Earth mass
        equilibrium_temp=280,
        h2o_abundance=-4.0,
        co2_abundance=-3.4,  # ~400 ppm
        ch4_abundance=-5.7,  # ~2 ppm
        o2_abundance=-0.7,   # 21%
        o3_abundance=-6.0,
        na_abundance=-10.0,  # Negligible
        cloud_fraction=0.6,
        mean_molecular_weight=28.97,
    )
    return stellar, planet


def create_ultra_hot_jupiter() -> Tuple[StellarParams, PlanetaryParams]:
    """Create parameters for an ultra-hot Jupiter."""
    stellar = StellarParams(
        temperature=7500,
        radius=1.8,
        spectral_type="A5V",
    )
    planet = PlanetaryParams(
        name="Ultra-Hot-Jupiter-1b",
        radius=1.9,
        mass=2.5,
        equilibrium_temp=2800,
        h2o_abundance=-4.0,  # Thermally dissociated
        co2_abundance=-6.0,
        ch4_abundance=-10.0, # Destroyed
        co_abundance=-3.0,
        na_abundance=-4.5,
        k_abundance=-5.0,
        tio_abundance=-5.5,
        vo_abundance=-6.0,
        feh_abundance=-5.5,
        cloud_fraction=0.0,  # Too hot for clouds
    )
    return stellar, planet


# =============================================================================
# Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic synthetic exoplanet transmission spectra"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./data/synthetic",
        help="Output directory for generated spectra"
    )
    parser.add_argument(
        "--n-spectra", type=int, default=10,
        help="Number of spectra to generate"
    )
    parser.add_argument(
        "--preset", type=str, default="random",
        choices=["random", "hot_jupiter", "warm_neptune", "earth_like", "ultra_hot"],
        help="Planet type preset"
    )
    parser.add_argument(
        "--wavelength-min", type=float, default=0.3,
        help="Minimum wavelength (μm)"
    )
    parser.add_argument(
        "--wavelength-max", type=float, default=15.0,
        help="Maximum wavelength (μm)"
    )
    parser.add_argument(
        "--resolution", type=int, default=2000,
        help="Number of wavelength points"
    )
    parser.add_argument(
        "--snr", type=float, default=50.0,
        help="Signal-to-noise ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Preset selection
    presets = {
        "hot_jupiter": create_hot_jupiter,
        "warm_neptune": create_warm_neptune,
        "earth_like": create_earth_like,
        "ultra_hot": create_ultra_hot_jupiter,
    }
    
    print(f"Generating {args.n_spectra} synthetic transmission spectra...")
    print(f"Wavelength range: {args.wavelength_min} - {args.wavelength_max} μm")
    print(f"Resolution: {args.resolution} points")
    print(f"Target SNR: {args.snr}")
    print(f"Output directory: {output_path}")
    print("-" * 50)
    
    all_spectra = []
    
    for i in range(args.n_spectra):
        # Select or randomize parameters
        if args.preset == "random":
            preset_choice = np.random.choice(list(presets.keys()))
            stellar, planet = presets[preset_choice]()
            
            # Add random variations
            planet.h2o_abundance += np.random.normal(0, 0.5)
            planet.co2_abundance += np.random.normal(0, 0.5)
            planet.equilibrium_temp *= np.random.uniform(0.9, 1.1)
        else:
            stellar, planet = presets[args.preset]()
        
        # Update planet name
        planet.name = f"Synthetic-{i+1:04d}b"
        
        # Set observation parameters
        obs = ObservationParams(
            wavelength_min=args.wavelength_min,
            wavelength_max=args.wavelength_max,
            resolution=args.resolution,
            snr=args.snr * np.random.uniform(0.7, 1.3),
        )
        
        # Generate spectrum
        df, metadata = generate_spectrum(stellar, planet, obs, seed=args.seed)
        
        # Save individual spectrum
        name = f"spectrum_{i+1:04d}"
        csv_path, json_path = save_spectrum(df, metadata, output_path, name)
        
        all_spectra.append({
            "name": planet.name,
            "csv_path": str(csv_path),
            "metadata": metadata,
        })
        
        print(f"  [{i+1}/{args.n_spectra}] Generated {planet.name}")
    
    # Save summary
    summary_path = output_path / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_spectra": args.n_spectra,
            "parameters": vars(args),
            "spectra": all_spectra,
            "generated_at": datetime.now().isoformat(),
        }, f, indent=2)
    
    print("-" * 50)
    print(f"Generation complete! Files saved to: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
