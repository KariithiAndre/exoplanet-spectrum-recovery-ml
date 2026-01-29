#!/usr/bin/env python
"""
Generate synthetic transmission spectra for training.

This script creates training data by simulating exoplanet transmission
spectra with various atmospheric compositions and noise levels.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


# Molecular absorption feature parameters (wavelength in microns)
MOLECULAR_FEATURES = {
    "H2O": [(1.15, 0.08), (1.4, 0.12), (1.9, 0.15), (2.7, 0.2)],
    "CO2": [(2.0, 0.1), (4.3, 0.25)],
    "CH4": [(2.3, 0.1), (3.3, 0.15)],
    "CO": [(4.6, 0.15)],
    "NH3": [(1.5, 0.1), (2.0, 0.12)],
    "Na": [(0.589, 0.02)],
    "K": [(0.766, 0.025)],
}


def gaussian_feature(
    wavelength: np.ndarray,
    center: float,
    width: float,
    depth: float,
) -> np.ndarray:
    """Generate a Gaussian absorption feature."""
    return depth * np.exp(-((wavelength - center) ** 2) / (2 * width ** 2))


def generate_spectrum(
    wavelength: np.ndarray,
    molecules: List[str],
    abundances: Dict[str, float],
    base_depth: float = 100.0,
    temperature: float = 1000.0,
) -> np.ndarray:
    """
    Generate a synthetic transmission spectrum.
    
    Args:
        wavelength: Wavelength grid in microns
        molecules: List of molecules to include
        abundances: Dictionary of log10 mixing ratios
        base_depth: Baseline transit depth in ppm
        temperature: Atmospheric temperature (affects feature widths)
        
    Returns:
        Transmission spectrum in ppm
    """
    spectrum = np.ones_like(wavelength) * base_depth
    
    # Temperature scaling for feature widths
    temp_scale = np.sqrt(temperature / 1000.0)
    
    for molecule in molecules:
        if molecule in MOLECULAR_FEATURES:
            abundance = 10 ** abundances.get(molecule, -4)
            
            for center, base_width in MOLECULAR_FEATURES[molecule]:
                if wavelength.min() <= center <= wavelength.max():
                    # Feature depth scales with abundance
                    depth = abundance * 1e4 * np.random.uniform(0.5, 1.5)
                    depth = np.clip(depth, 1, 50)
                    
                    # Width scales with temperature
                    width = base_width * temp_scale * np.random.uniform(0.8, 1.2)
                    
                    spectrum += gaussian_feature(wavelength, center, width, depth)
    
    return spectrum


def add_realistic_noise(
    spectrum: np.ndarray,
    snr: float,
    systematic_amplitude: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic noise to spectrum.
    
    Includes:
    - Photon noise (Gaussian)
    - Optional systematic trends
    
    Returns:
        Noisy spectrum and error array
    """
    # Base noise level
    noise_std = spectrum.std() / snr
    photon_noise = np.random.normal(0, noise_std, len(spectrum))
    
    # Optional systematic trend
    if systematic_amplitude > 0:
        # Simple polynomial trend
        x = np.linspace(-1, 1, len(spectrum))
        coeffs = np.random.uniform(-1, 1, 3) * systematic_amplitude
        systematic = np.polyval(coeffs, x)
    else:
        systematic = 0
    
    noisy_spectrum = spectrum + photon_noise + systematic
    error = np.full_like(spectrum, noise_std)
    
    return noisy_spectrum, error


def main(args):
    """Generate synthetic dataset."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wavelength grid
    wavelength = np.linspace(args.wl_min, args.wl_max, args.n_wavelengths)
    
    # Initialize arrays
    clean_spectra = np.zeros((args.n_samples, args.n_wavelengths), dtype=np.float32)
    noisy_spectra = np.zeros((args.n_samples, args.n_wavelengths), dtype=np.float32)
    errors = np.zeros((args.n_samples, args.n_wavelengths), dtype=np.float32)
    metadata = []
    
    molecules = list(MOLECULAR_FEATURES.keys())
    
    for i in tqdm(range(args.n_samples), desc="Generating spectra"):
        # Random atmospheric parameters
        n_molecules = np.random.randint(2, 6)
        selected_molecules = np.random.choice(molecules, n_molecules, replace=False).tolist()
        
        abundances = {
            mol: np.random.uniform(-6, -2) for mol in selected_molecules
        }
        
        temperature = np.random.uniform(500, 2500)
        base_depth = np.random.uniform(50, 200)
        snr = np.random.uniform(args.min_snr, args.max_snr)
        
        # Generate spectrum
        clean = generate_spectrum(
            wavelength, selected_molecules, abundances, base_depth, temperature
        )
        noisy, error = add_realistic_noise(clean, snr, args.systematic_amplitude)
        
        clean_spectra[i] = clean
        noisy_spectra[i] = noisy
        errors[i] = error
        
        metadata.append({
            "index": i,
            "molecules": selected_molecules,
            "abundances": abundances,
            "temperature": temperature,
            "base_depth": base_depth,
            "snr": snr,
        })
    
    # Save data
    np.save(output_dir / "wavelength.npy", wavelength)
    np.save(output_dir / "clean_spectra.npy", clean_spectra)
    np.save(output_dir / "noisy_spectra.npy", noisy_spectra)
    np.save(output_dir / "errors.npy", errors)
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {args.n_samples} spectra:")
    print(f"  Wavelength range: {args.wl_min} - {args.wl_max} μm")
    print(f"  Wavelength points: {args.n_wavelengths}")
    print(f"  SNR range: {args.min_snr} - {args.max_snr}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    
    parser.add_argument("--output-dir", type=str, default="./data/synthetic",
                        help="Output directory")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of spectra to generate")
    parser.add_argument("--n-wavelengths", type=int, default=1024,
                        help="Number of wavelength points")
    parser.add_argument("--wl-min", type=float, default=0.3,
                        help="Minimum wavelength (microns)")
    parser.add_argument("--wl-max", type=float, default=15.0,
                        help="Maximum wavelength (microns)")
    parser.add_argument("--min-snr", type=float, default=5,
                        help="Minimum SNR")
    parser.add_argument("--max-snr", type=float, default=100,
                        help="Maximum SNR")
    parser.add_argument("--systematic-amplitude", type=float, default=0.0,
                        help="Amplitude of systematic noise")
    
    args = parser.parse_args()
    main(args)
