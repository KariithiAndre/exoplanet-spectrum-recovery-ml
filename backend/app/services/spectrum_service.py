"""Service for spectrum processing and analysis."""

import numpy as np
from typing import Optional, Tuple
import json
import io
from datetime import datetime

from app.models.spectrum import SpectrumData, AnalysisResult


class SpectrumService:
    """Service class for spectrum processing operations."""
    
    async def process_file(self, content: bytes, file_ext: str) -> SpectrumData:
        """
        Process uploaded spectrum file and extract data.
        
        Args:
            content: Raw file content
            file_ext: File extension (.fits, .csv, .json)
            
        Returns:
            SpectrumData object with wavelength and flux arrays
        """
        if file_ext in [".fits", ".fit"]:
            return await self._process_fits(content)
        elif file_ext == ".csv":
            return await self._process_csv(content)
        elif file_ext == ".json":
            return await self._process_json(content)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    async def _process_fits(self, content: bytes) -> SpectrumData:
        """Process FITS file format."""
        try:
            from astropy.io import fits
            
            with io.BytesIO(content) as f:
                with fits.open(f) as hdul:
                    # Try to find spectrum data in common HDU structures
                    data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                    
                    # Extract wavelength and flux columns
                    if hasattr(data, 'names'):
                        # Table data
                        wavelength = data['WAVELENGTH'].tolist()
                        flux = data['FLUX'].tolist() if 'FLUX' in data.names else data['SPEC'].tolist()
                        error = data['ERROR'].tolist() if 'ERROR' in data.names else None
                    else:
                        # Image data - assume first row is wavelength, second is flux
                        wavelength = data[0].tolist()
                        flux = data[1].tolist()
                        error = data[2].tolist() if len(data) > 2 else None
                    
                    return SpectrumData(
                        wavelength=wavelength,
                        flux=flux,
                        error=error,
                    )
        except ImportError:
            raise ValueError("astropy is required for FITS file processing")
        except Exception as e:
            raise ValueError(f"Error processing FITS file: {str(e)}")
    
    async def _process_csv(self, content: bytes) -> SpectrumData:
        """Process CSV file format."""
        import pandas as pd
        
        df = pd.read_csv(io.BytesIO(content))
        
        # Try common column name patterns
        wavelength_cols = ['wavelength', 'wave', 'lambda', 'wl', 'WAVELENGTH']
        flux_cols = ['flux', 'transit_depth', 'depth', 'spec', 'FLUX']
        error_cols = ['error', 'err', 'uncertainty', 'ERROR']
        
        wavelength = None
        flux = None
        error = None
        
        for col in wavelength_cols:
            if col in df.columns:
                wavelength = df[col].tolist()
                break
        
        for col in flux_cols:
            if col in df.columns:
                flux = df[col].tolist()
                break
        
        for col in error_cols:
            if col in df.columns:
                error = df[col].tolist()
                break
        
        if wavelength is None or flux is None:
            # Fall back to positional columns
            wavelength = df.iloc[:, 0].tolist()
            flux = df.iloc[:, 1].tolist()
            if df.shape[1] > 2:
                error = df.iloc[:, 2].tolist()
        
        return SpectrumData(
            wavelength=wavelength,
            flux=flux,
            error=error,
        )
    
    async def _process_json(self, content: bytes) -> SpectrumData:
        """Process JSON file format."""
        data = json.loads(content.decode('utf-8'))
        
        return SpectrumData(
            wavelength=data.get('wavelength', data.get('wave', [])),
            flux=data.get('flux', data.get('transit_depth', [])),
            error=data.get('error', data.get('uncertainty')),
        )
    
    async def analyze(self, spectrum: SpectrumData) -> AnalysisResult:
        """
        Perform initial analysis on spectrum data.
        
        Args:
            spectrum: SpectrumData object
            
        Returns:
            AnalysisResult with SNR, confidence, and detected features
        """
        import time
        start_time = time.time()
        
        flux = np.array(spectrum.flux)
        error = np.array(spectrum.error) if spectrum.error else np.ones_like(flux) * np.std(flux)
        
        # Calculate SNR
        snr = float(np.median(flux / error))
        
        # Mock feature detection (in production, use ML model)
        features = self._detect_features(spectrum)
        
        # Calculate confidence based on SNR
        confidence = min(0.99, snr / 100.0)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return AnalysisResult(
            snr=round(snr, 1),
            confidence=round(confidence, 2),
            processing_time=processing_time,
            features=features,
        )
    
    def _detect_features(self, spectrum: SpectrumData) -> list:
        """Detect molecular absorption features in spectrum."""
        # Mock feature detection - in production, use trained model
        features = []
        
        wavelength = np.array(spectrum.wavelength)
        flux = np.array(spectrum.flux)
        
        # Known molecular feature wavelengths (microns)
        molecular_features = {
            "H2O": [1.15, 1.4, 1.9],
            "CO2": [2.0, 4.3],
            "CH4": [2.3, 3.3],
            "CO": [4.6],
            "Na": [0.589],
            "K": [0.766],
        }
        
        for molecule, wavelengths in molecular_features.items():
            for w in wavelengths:
                if wavelength.min() <= w <= wavelength.max():
                    # Check for absorption feature at this wavelength
                    idx = np.argmin(np.abs(wavelength - w))
                    local_flux = flux[max(0, idx-5):min(len(flux), idx+5)]
                    if len(local_flux) > 0:
                        # Simple significance estimate
                        significance = abs(flux[idx] - np.median(flux)) / np.std(flux)
                        if significance > 1.5:
                            features.append({
                                "molecule": molecule,
                                "wavelength": w,
                                "significance": round(significance, 1),
                            })
        
        return features
    
    async def recover(
        self,
        spectrum_id: str,
        model_id: str,
        wavelength_range: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """
        Apply ML model to recover spectrum.
        
        This is a placeholder - in production, load and run the actual model.
        """
        # TODO: Implement actual model inference
        return {
            "status": "completed",
            "spectrum_id": spectrum_id,
            "model_id": model_id,
            "message": "Recovery completed successfully",
        }
