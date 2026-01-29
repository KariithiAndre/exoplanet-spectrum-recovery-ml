"""
Model Explainability Tools for Spectral Classification

Provides interpretability methods for understanding model predictions:
1. Saliency maps (gradient-based wavelength importance)
2. Integrated Gradients for attribution
3. Attention visualization for Transformers
4. Influential absorption region detection
5. Visual interpretation plots

Author: Exoplanet Spectrum Recovery Project
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

MOLECULE_NAMES = [
    "H2O", "CO2", "CO", "CH4", "NH3", "O2", "O3",
    "Na", "K", "TiO", "VO", "FeH", "H2S", "HCN"
]

# Molecular absorption band reference (wavelength in microns)
MOLECULAR_BANDS = {
    "H2O": [(1.4, 1.5), (1.8, 2.0), (2.6, 2.8), (5.5, 7.5)],
    "CO2": [(2.0, 2.1), (2.7, 2.8), (4.2, 4.4), (15.0, 15.5)],
    "CO": [(2.3, 2.4), (4.5, 4.8)],
    "CH4": [(1.65, 1.75), (2.2, 2.4), (3.2, 3.5), (7.5, 8.0)],
    "NH3": [(1.5, 1.55), (2.0, 2.1), (2.9, 3.1), (10.0, 11.0)],
    "O2": [(0.76, 0.77), (1.27, 1.28)],
    "O3": [(9.0, 10.0), (0.5, 0.7)],
    "Na": [(0.589, 0.590)],
    "K": [(0.766, 0.770)],
    "TiO": [(0.7, 0.9)],
    "VO": [(0.7, 0.85)],
    "FeH": [(0.99, 1.0), (1.2, 1.25)],
    "H2S": [(2.5, 2.6), (3.8, 4.0)],
    "HCN": [(3.0, 3.1), (14.0, 14.5)],
}

# Custom colormap for saliency
SALIENCY_COLORS = [
    (0.0, 0.0, 0.5),   # Dark blue (negative)
    (0.0, 0.5, 1.0),   # Light blue
    (1.0, 1.0, 1.0),   # White (neutral)
    (1.0, 0.5, 0.0),   # Orange
    (0.8, 0.0, 0.0),   # Dark red (positive)
]


# =============================================================================
# Configuration
# =============================================================================

class ExplainMethod(Enum):
    """Available explanation methods."""
    GRADIENT = "gradient"
    SMOOTH_GRAD = "smooth_grad"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRAD_CAM = "grad_cam"
    ATTENTION = "attention"
    OCCLUSION = "occlusion"
    GUIDED_BACKPROP = "guided_backprop"


@dataclass
class ExplainConfig:
    """Configuration for explainability methods."""
    
    # Method selection
    method: ExplainMethod = ExplainMethod.INTEGRATED_GRADIENTS
    
    # Integrated Gradients
    ig_steps: int = 50
    ig_baseline: str = "zero"  # "zero", "mean", "noise"
    
    # SmoothGrad
    smooth_samples: int = 50
    smooth_noise_level: float = 0.15
    
    # Occlusion
    occlusion_window: int = 10
    occlusion_stride: int = 5
    
    # Visualization
    colormap: str = "RdBu_r"
    highlight_threshold: float = 0.1
    show_molecule_bands: bool = True
    
    # Output
    figsize: Tuple[int, int] = (14, 10)
    dpi: int = 150


# =============================================================================
# Saliency Methods
# =============================================================================

class GradientSaliency:
    """
    Compute gradient-based saliency maps.
    
    Shows which wavelengths most influence the model's predictions.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
    
    def compute(
        self,
        spectrum: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
        target_output: str = "molecules",
        target_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute gradient saliency for input spectrum.
        
        Args:
            spectrum: Input spectrum (length,) or (1, length)
            wavelengths: Optional wavelength values
            target_output: Which output to explain ("molecules", "planet_class", "habitability")
            target_index: Index of specific class/molecule (None for sum)
            
        Returns:
            Saliency values for each wavelength position
        """
        self.model.eval()
        
        # Prepare input
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        
        spectrum = spectrum.to(self.device).requires_grad_(True)
        
        if wavelengths is not None:
            wavelengths = wavelengths.to(self.device)
        
        # Forward pass
        outputs = self.model(spectrum, wavelengths)
        
        # Select target
        target = outputs[target_output]
        
        if target_index is not None:
            if target.dim() > 1:
                target = target[:, target_index]
            score = target.sum()
        else:
            score = target.sum()
        
        # Backward pass
        score.backward()
        
        # Get gradients
        saliency = spectrum.grad.abs().squeeze().cpu().numpy()
        
        return saliency
    
    def compute_smooth_grad(
        self,
        spectrum: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
        target_output: str = "molecules",
        target_index: Optional[int] = None,
        n_samples: int = 50,
        noise_level: float = 0.15,
    ) -> np.ndarray:
        """
        Compute SmoothGrad saliency (averaged over noisy inputs).
        """
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        
        std = noise_level * (spectrum.max() - spectrum.min()).item()
        
        saliencies = []
        for _ in range(n_samples):
            noisy = spectrum + torch.randn_like(spectrum) * std
            sal = self.compute(noisy, wavelengths, target_output, target_index)
            saliencies.append(sal)
        
        return np.mean(saliencies, axis=0)


class IntegratedGradients:
    """
    Integrated Gradients for attribution.
    
    Provides theoretically grounded attribution by integrating gradients
    along a path from baseline to input.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
    
    def compute(
        self,
        spectrum: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
        target_output: str = "molecules",
        target_index: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            spectrum: Input spectrum
            wavelengths: Optional wavelength values
            target_output: Which output to explain
            target_index: Specific class index
            baseline: Baseline input (default: zeros)
            steps: Number of integration steps
            
        Returns:
            Attribution values for each wavelength
        """
        self.model.eval()
        
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        
        spectrum = spectrum.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(spectrum)
        else:
            baseline = baseline.to(self.device)
        
        if wavelengths is not None:
            wavelengths = wavelengths.to(self.device)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=self.device)
        
        # Compute gradients at each step
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (spectrum - baseline)
            interpolated.requires_grad_(True)
            
            outputs = self.model(interpolated, wavelengths)
            target = outputs[target_output]
            
            if target_index is not None and target.dim() > 1:
                target = target[:, target_index]
            
            score = target.sum()
            score.backward()
            
            gradients.append(interpolated.grad.clone())
            self.model.zero_grad()
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Scale by input - baseline
        attributions = (spectrum - baseline) * avg_gradients
        
        return attributions.squeeze().cpu().numpy()
    
    def compute_convergence_delta(
        self,
        spectrum: torch.Tensor,
        attributions: np.ndarray,
        wavelengths: Optional[torch.Tensor] = None,
        target_output: str = "molecules",
        target_index: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute convergence delta (completeness axiom check).
        
        Should be close to 0 if attributions are accurate.
        """
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        
        spectrum = spectrum.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(spectrum)
        else:
            baseline = baseline.to(self.device)
        
        if wavelengths is not None:
            wavelengths = wavelengths.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            out_input = self.model(spectrum, wavelengths)
            out_baseline = self.model(baseline, wavelengths)
            
            target_input = out_input[target_output]
            target_baseline = out_baseline[target_output]
            
            if target_index is not None and target_input.dim() > 1:
                target_input = target_input[:, target_index]
                target_baseline = target_baseline[:, target_index]
            
            output_diff = (target_input - target_baseline).sum().item()
        
        attribution_sum = attributions.sum()
        
        return abs(output_diff - attribution_sum)


class OcclusionAnalysis:
    """
    Occlusion-based importance analysis.
    
    Measures importance by masking regions and observing prediction changes.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
    
    def compute(
        self,
        spectrum: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
        target_output: str = "molecules",
        target_index: Optional[int] = None,
        window_size: int = 10,
        stride: int = 5,
        mask_value: float = 0.0,
    ) -> np.ndarray:
        """
        Compute occlusion-based importance.
        """
        self.model.eval()
        
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        
        spectrum = spectrum.to(self.device)
        seq_len = spectrum.size(-1)
        
        if wavelengths is not None:
            wavelengths = wavelengths.to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_out = self.model(spectrum, wavelengths)
            baseline_score = self._get_score(baseline_out, target_output, target_index)
        
        # Compute importance at each position
        importance = np.zeros(seq_len)
        counts = np.zeros(seq_len)
        
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            
            # Create masked input
            masked = spectrum.clone()
            masked[:, start:end] = mask_value
            
            with torch.no_grad():
                masked_out = self.model(masked, wavelengths)
                masked_score = self._get_score(masked_out, target_output, target_index)
            
            # Importance = drop in score when masked
            drop = baseline_score - masked_score
            importance[start:end] += drop
            counts[start:end] += 1
        
        # Average overlapping regions
        importance = importance / np.maximum(counts, 1)
        
        return importance
    
    def _get_score(
        self,
        outputs: Dict[str, torch.Tensor],
        target_output: str,
        target_index: Optional[int],
    ) -> float:
        target = outputs[target_output]
        
        if target_index is not None and target.dim() > 1:
            target = target[:, target_index]
        
        return target.sum().item()


class AttentionAnalysis:
    """
    Attention weight visualization for Transformer models.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self._attention_weights = []
        self._hooks = []
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        self._attention_weights = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # Some attention modules return (output, attention_weights)
                self._attention_weights.append(output[1].detach())
        
        # Find attention modules
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if hasattr(module, 'forward'):
                    handle = module.register_forward_hook(hook_fn)
                    self._hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def compute(
        self,
        spectrum: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
        layer: int = -1,
        head: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract attention weights from model.
        
        Args:
            spectrum: Input spectrum
            wavelengths: Optional wavelength values
            layer: Which layer's attention to extract (-1 for last)
            head: Specific attention head (None for average)
            
        Returns:
            Attention weights or averaged importance
        """
        self.model.eval()
        
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        
        spectrum = spectrum.to(self.device)
        
        if wavelengths is not None:
            wavelengths = wavelengths.to(self.device)
        
        self._register_hooks()
        
        try:
            with torch.no_grad():
                _ = self.model(spectrum, wavelengths)
        finally:
            self._remove_hooks()
        
        if not self._attention_weights:
            logger.warning("No attention weights captured. Model may not have attention modules.")
            return np.ones(spectrum.size(-1)) / spectrum.size(-1)
        
        # Get specified layer
        attn = self._attention_weights[layer]  # (batch, heads, seq, seq)
        
        if head is not None:
            attn = attn[:, head]  # (batch, seq, seq)
        else:
            attn = attn.mean(dim=1)  # Average over heads
        
        # Get importance per position (sum of attention received)
        importance = attn.sum(dim=1).squeeze().cpu().numpy()
        
        return importance


# =============================================================================
# Region Detection
# =============================================================================

class InfluentialRegionDetector:
    """
    Detect influential absorption regions in spectra.
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        min_region_width: int = 3,
        merge_distance: int = 5,
    ):
        self.threshold = threshold
        self.min_region_width = min_region_width
        self.merge_distance = merge_distance
    
    def detect(
        self,
        saliency: np.ndarray,
        wavelengths: np.ndarray,
        mode: str = "positive",  # "positive", "negative", "both"
    ) -> List[Dict[str, Any]]:
        """
        Detect influential regions from saliency map.
        
        Args:
            saliency: Saliency values
            wavelengths: Wavelength array
            mode: Which regions to detect
            
        Returns:
            List of region dictionaries with start, end, wavelengths, importance
        """
        # Normalize saliency
        saliency_norm = saliency / (np.abs(saliency).max() + 1e-8)
        
        # Find regions above threshold
        if mode == "positive":
            mask = saliency_norm > self.threshold
        elif mode == "negative":
            mask = saliency_norm < -self.threshold
        else:
            mask = np.abs(saliency_norm) > self.threshold
        
        # Find connected regions
        regions = self._find_connected_regions(mask)
        
        # Filter by minimum width and merge close regions
        regions = self._filter_and_merge(regions)
        
        # Build region info
        result = []
        for start, end in regions:
            region = {
                'start_idx': start,
                'end_idx': end,
                'start_wavelength': wavelengths[start],
                'end_wavelength': wavelengths[end],
                'center_wavelength': wavelengths[(start + end) // 2],
                'width': wavelengths[end] - wavelengths[start],
                'importance': saliency[start:end + 1].mean(),
                'max_importance': saliency[start:end + 1].max(),
                'peak_idx': start + np.argmax(np.abs(saliency[start:end + 1])),
            }
            region['peak_wavelength'] = wavelengths[region['peak_idx']]
            result.append(region)
        
        # Sort by importance
        result.sort(key=lambda x: abs(x['max_importance']), reverse=True)
        
        return result
    
    def _find_connected_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find connected regions in boolean mask."""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i - 1))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask) - 1))
        
        return regions
    
    def _filter_and_merge(
        self,
        regions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Filter by width and merge close regions."""
        # Filter by minimum width
        regions = [(s, e) for s, e in regions if e - s + 1 >= self.min_region_width]
        
        if not regions:
            return []
        
        # Merge close regions
        merged = [regions[0]]
        
        for start, end in regions[1:]:
            last_start, last_end = merged[-1]
            
            if start - last_end <= self.merge_distance:
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))
        
        return merged
    
    def match_molecules(
        self,
        regions: List[Dict[str, Any]],
        tolerance: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Match detected regions with known molecular absorption bands.
        """
        for region in regions:
            center = region['center_wavelength']
            matches = []
            
            for molecule, bands in MOLECULAR_BANDS.items():
                for band_start, band_end in bands:
                    # Check if region overlaps with known band
                    if (region['start_wavelength'] - tolerance <= band_end and
                        region['end_wavelength'] + tolerance >= band_start):
                        
                        overlap = min(region['end_wavelength'], band_end) - \
                                  max(region['start_wavelength'], band_start)
                        band_width = band_end - band_start
                        confidence = max(0, overlap / band_width)
                        
                        matches.append({
                            'molecule': molecule,
                            'band': (band_start, band_end),
                            'confidence': min(confidence, 1.0),
                        })
            
            region['molecule_matches'] = matches
        
        return regions


# =============================================================================
# Visualization
# =============================================================================

class ExplainabilityVisualizer:
    """
    Create visual interpretation plots for model predictions.
    """
    
    def __init__(self, config: Optional[ExplainConfig] = None):
        self.config = config or ExplainConfig()
        self._setup_colormap()
    
    def _setup_colormap(self):
        """Create custom saliency colormap."""
        self.saliency_cmap = LinearSegmentedColormap.from_list(
            'saliency', SALIENCY_COLORS, N=256
        )
    
    def plot_saliency_map(
        self,
        spectrum: np.ndarray,
        saliency: np.ndarray,
        wavelengths: np.ndarray,
        title: str = "Wavelength Importance (Saliency Map)",
        save_path: Optional[str] = None,
        show_molecules: bool = True,
    ) -> plt.Figure:
        """
        Plot spectrum with saliency overlay.
        """
        fig, axes = plt.subplots(3, 1, figsize=self.config.figsize, 
                                  gridspec_kw={'height_ratios': [2, 1, 0.3]})
        
        # Normalize saliency for visualization
        saliency_norm = saliency / (np.abs(saliency).max() + 1e-8)
        
        # Top: Spectrum with saliency coloring
        ax1 = axes[0]
        
        # Create colored line segments
        points = np.array([wavelengths, spectrum]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        
        norm = plt.Normalize(-1, 1)
        lc = LineCollection(segments, cmap=self.saliency_cmap, norm=norm)
        lc.set_array(saliency_norm[:-1])
        lc.set_linewidth(2)
        
        ax1.add_collection(lc)
        ax1.set_xlim(wavelengths.min(), wavelengths.max())
        ax1.set_ylim(spectrum.min() * 0.95, spectrum.max() * 1.05)
        ax1.set_ylabel('Flux / Transit Depth', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add molecular band markers
        if show_molecules:
            self._add_molecule_bands(ax1, wavelengths)
        
        # Middle: Saliency bar plot
        ax2 = axes[1]
        
        colors = self.saliency_cmap(norm(saliency_norm))
        ax2.bar(wavelengths, saliency_norm, width=np.diff(wavelengths).mean(),
                color=colors, edgecolor='none')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.axhline(y=self.config.highlight_threshold, color='gray', 
                    linestyle='--', alpha=0.5)
        ax2.axhline(y=-self.config.highlight_threshold, color='gray',
                    linestyle='--', alpha=0.5)
        ax2.set_xlim(wavelengths.min(), wavelengths.max())
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel('Wavelength (μm)', fontsize=12)
        ax2.set_ylabel('Importance', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Bottom: Colorbar
        ax3 = axes[2]
        sm = plt.cm.ScalarMappable(cmap=self.saliency_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax3, orientation='horizontal')
        cbar.set_label('Wavelength Importance (Saliency)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_influential_regions(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        regions: List[Dict[str, Any]],
        title: str = "Influential Absorption Regions",
        save_path: Optional[str] = None,
        top_n: int = 10,
    ) -> plt.Figure:
        """
        Highlight influential absorption regions on spectrum.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot spectrum
        ax.plot(wavelengths, spectrum, 'b-', linewidth=1.5, label='Spectrum')
        
        # Highlight regions
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, min(top_n, len(regions))))
        
        for i, region in enumerate(regions[:top_n]):
            start_wl = region['start_wavelength']
            end_wl = region['end_wavelength']
            
            # Highlight region
            ax.axvspan(start_wl, end_wl, alpha=0.3, color=colors[i],
                       label=f"Region {i+1}: {start_wl:.2f}-{end_wl:.2f} μm")
            
            # Mark peak
            peak_idx = region['peak_idx']
            ax.scatter(wavelengths[peak_idx], spectrum[peak_idx], 
                       color=colors[i], s=50, zorder=5)
            
            # Add molecule labels if matched
            if 'molecule_matches' in region and region['molecule_matches']:
                top_match = region['molecule_matches'][0]
                if top_match['confidence'] > 0.3:
                    ax.annotate(
                        top_match['molecule'],
                        xy=(region['center_wavelength'], spectrum[region['peak_idx']]),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        color=colors[i],
                    )
        
        ax.set_xlabel('Wavelength (μm)', fontsize=12)
        ax.set_ylabel('Flux / Transit Depth', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_explanation(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        prediction: Dict[str, Any],
        saliencies: Dict[str, np.ndarray],
        title: str = "Prediction Explanation",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive explanation plot for a prediction.
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Layout: 3 rows
        # Row 1: Spectrum with overall saliency
        # Row 2: Per-molecule saliencies (top 4)
        # Row 3: Prediction summary
        
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1.5, 1], hspace=0.3, wspace=0.3)
        
        # Row 1: Main spectrum
        ax_main = fig.add_subplot(gs[0, :])
        
        overall_saliency = saliencies.get('overall', np.zeros_like(spectrum))
        saliency_norm = overall_saliency / (np.abs(overall_saliency).max() + 1e-8)
        
        # Plot with colored background
        ax_main.fill_between(wavelengths, spectrum.min(), spectrum,
                             where=saliency_norm > self.config.highlight_threshold,
                             alpha=0.3, color='red', label='High importance')
        ax_main.fill_between(wavelengths, spectrum.min(), spectrum,
                             where=saliency_norm < -self.config.highlight_threshold,
                             alpha=0.3, color='blue', label='Low importance')
        ax_main.plot(wavelengths, spectrum, 'k-', linewidth=1.5)
        
        ax_main.set_xlabel('Wavelength (μm)', fontsize=11)
        ax_main.set_ylabel('Flux', fontsize=11)
        ax_main.set_title(title, fontsize=14, fontweight='bold')
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3)
        
        # Row 2: Per-molecule saliencies
        detected_molecules = prediction.get('detected_molecules', [])[:4]
        
        for i, mol in enumerate(detected_molecules):
            ax = fig.add_subplot(gs[1, i])
            
            mol_saliency = saliencies.get(mol, np.zeros_like(spectrum))
            sal_norm = mol_saliency / (np.abs(mol_saliency).max() + 1e-8)
            
            # Color by saliency
            for j in range(len(wavelengths) - 1):
                color = self.saliency_cmap((sal_norm[j] + 1) / 2)
                ax.fill_between(wavelengths[j:j+2], 0, sal_norm[j:j+2],
                                color=color, alpha=0.7)
            
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_xlim(wavelengths.min(), wavelengths.max())
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel('Wavelength (μm)', fontsize=9)
            ax.set_title(f'{mol} Attribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Mark known bands
            if mol in MOLECULAR_BANDS:
                for band_start, band_end in MOLECULAR_BANDS[mol]:
                    if band_start >= wavelengths.min() and band_end <= wavelengths.max():
                        ax.axvspan(band_start, band_end, alpha=0.2, color='green')
        
        # Row 3: Prediction summary
        ax_summary = fig.add_subplot(gs[2, :2])
        ax_summary.axis('off')
        
        # Build summary text
        summary = f"Detected Molecules: {', '.join(prediction.get('detected_molecules', ['None']))}\n"
        summary += f"Planet Class: {prediction.get('planet_class', 'Unknown')}\n"
        summary += f"Habitability Score: {prediction.get('habitability_score', 0):.3f}"
        
        if 'habitability_uncertainty' in prediction:
            summary += f" ± {prediction['habitability_uncertainty']:.3f}"
        
        ax_summary.text(0.05, 0.5, summary, transform=ax_summary.transAxes,
                        fontsize=12, verticalalignment='center',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Molecule probabilities bar chart
        ax_probs = fig.add_subplot(gs[2, 2:])
        
        mol_probs = prediction.get('molecule_probabilities', {})
        if mol_probs:
            molecules = list(mol_probs.keys())[:10]
            probs = [mol_probs[m] for m in molecules]
            
            colors = ['green' if p > 0.5 else 'gray' for p in probs]
            bars = ax_probs.barh(molecules, probs, color=colors, alpha=0.7)
            ax_probs.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
            ax_probs.set_xlim(0, 1)
            ax_probs.set_xlabel('Detection Probability', fontsize=10)
            ax_probs.set_title('Molecule Probabilities', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        wavelengths: np.ndarray,
        title: str = "Attention Weights",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot attention weight heatmap.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        
        # Add tick labels (subsample for readability)
        n_ticks = 10
        tick_indices = np.linspace(0, len(wavelengths) - 1, n_ticks, dtype=int)
        tick_labels = [f'{wavelengths[i]:.2f}' for i in tick_indices]
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels)
        
        ax.set_xlabel('Key Position (Wavelength μm)', fontsize=12)
        ax.set_ylabel('Query Position (Wavelength μm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_comparison(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        saliencies: Dict[str, np.ndarray],
        title: str = "Method Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare different explanation methods.
        """
        n_methods = len(saliencies)
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=(14, 3 * (n_methods + 1)),
                                  sharex=True)
        
        # Top: Original spectrum
        axes[0].plot(wavelengths, spectrum, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Flux', fontsize=10)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Each method
        for i, (method_name, saliency) in enumerate(saliencies.items(), 1):
            ax = axes[i]
            
            saliency_norm = saliency / (np.abs(saliency).max() + 1e-8)
            
            # Color-coded bar plot
            colors = self.saliency_cmap((saliency_norm + 1) / 2)
            ax.bar(wavelengths, saliency_norm, width=np.diff(wavelengths).mean(),
                   color=colors, edgecolor='none')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel(method_name, fontsize=10)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Wavelength (μm)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _add_molecule_bands(
        self,
        ax: plt.Axes,
        wavelengths: np.ndarray,
        alpha: float = 0.1,
    ):
        """Add molecular band reference markers."""
        wl_min, wl_max = wavelengths.min(), wavelengths.max()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(MOLECULAR_BANDS)))
        
        for (mol, bands), color in zip(MOLECULAR_BANDS.items(), colors):
            for band_start, band_end in bands:
                if band_start <= wl_max and band_end >= wl_min:
                    ax.axvspan(
                        max(band_start, wl_min),
                        min(band_end, wl_max),
                        alpha=alpha,
                        color=color,
                        label=mol if bands.index((band_start, band_end)) == 0 else None,
                    )


# =============================================================================
# Main Explainer Class
# =============================================================================

class SpectralExplainer:
    """
    Unified interface for model explainability.
    
    Example:
        explainer = SpectralExplainer(model)
        
        # Get saliency map
        saliency = explainer.explain(spectrum, wavelengths, method="integrated_gradients")
        
        # Detect influential regions
        regions = explainer.detect_regions(saliency, wavelengths)
        
        # Visualize
        explainer.visualize(spectrum, wavelengths, saliency, regions)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ExplainConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config or ExplainConfig()
        self.device = device or next(model.parameters()).device
        
        # Initialize methods
        self.gradient = GradientSaliency(model, device)
        self.integrated_gradients = IntegratedGradients(model, device)
        self.occlusion = OcclusionAnalysis(model, device)
        self.attention = AttentionAnalysis(model, device)
        
        # Region detector
        self.region_detector = InfluentialRegionDetector(
            threshold=self.config.highlight_threshold
        )
        
        # Visualizer
        self.visualizer = ExplainabilityVisualizer(self.config)
    
    def explain(
        self,
        spectrum: Union[np.ndarray, torch.Tensor],
        wavelengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        method: str = "integrated_gradients",
        target_output: str = "molecules",
        target_index: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate explanation for model prediction.
        
        Args:
            spectrum: Input spectrum
            wavelengths: Wavelength values
            method: Explanation method
            target_output: Which output to explain
            target_index: Specific class/molecule index
            **kwargs: Method-specific arguments
            
        Returns:
            Saliency/attribution values
        """
        # Convert to tensor
        if isinstance(spectrum, np.ndarray):
            spectrum = torch.from_numpy(spectrum).float()
        
        if wavelengths is not None and isinstance(wavelengths, np.ndarray):
            wavelengths = torch.from_numpy(wavelengths).float()
        
        # Select method
        if method == "gradient":
            return self.gradient.compute(
                spectrum, wavelengths, target_output, target_index
            )
        
        elif method == "smooth_grad":
            return self.gradient.compute_smooth_grad(
                spectrum, wavelengths, target_output, target_index,
                n_samples=kwargs.get('n_samples', self.config.smooth_samples),
                noise_level=kwargs.get('noise_level', self.config.smooth_noise_level),
            )
        
        elif method == "integrated_gradients":
            return self.integrated_gradients.compute(
                spectrum, wavelengths, target_output, target_index,
                baseline=kwargs.get('baseline'),
                steps=kwargs.get('steps', self.config.ig_steps),
            )
        
        elif method == "occlusion":
            return self.occlusion.compute(
                spectrum, wavelengths, target_output, target_index,
                window_size=kwargs.get('window_size', self.config.occlusion_window),
                stride=kwargs.get('stride', self.config.occlusion_stride),
            )
        
        elif method == "attention":
            return self.attention.compute(
                spectrum, wavelengths,
                layer=kwargs.get('layer', -1),
                head=kwargs.get('head'),
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def explain_all_molecules(
        self,
        spectrum: Union[np.ndarray, torch.Tensor],
        wavelengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        method: str = "integrated_gradients",
        molecules: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate explanations for all molecule detections.
        """
        molecules = molecules or MOLECULE_NAMES
        
        saliencies = {}
        
        for i, mol in enumerate(molecules):
            try:
                saliency = self.explain(
                    spectrum, wavelengths, method,
                    target_output="molecules", target_index=i
                )
                saliencies[mol] = saliency
            except Exception as e:
                logger.warning(f"Failed to compute saliency for {mol}: {e}")
        
        # Overall saliency (sum)
        if saliencies:
            saliencies['overall'] = np.mean(list(saliencies.values()), axis=0)
        
        return saliencies
    
    def detect_regions(
        self,
        saliency: np.ndarray,
        wavelengths: np.ndarray,
        match_molecules: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Detect influential regions from saliency map.
        """
        regions = self.region_detector.detect(saliency, wavelengths)
        
        if match_molecules:
            regions = self.region_detector.match_molecules(regions)
        
        return regions
    
    def visualize(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        saliency: np.ndarray,
        regions: Optional[List[Dict[str, Any]]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate all visualization plots.
        """
        figures = {}
        
        save_path = lambda name: str(Path(save_dir) / name) if save_dir else None
        
        # Saliency map
        figures['saliency'] = self.visualizer.plot_saliency_map(
            spectrum, saliency, wavelengths,
            save_path=save_path('saliency_map.png'),
        )
        
        # Influential regions
        if regions:
            figures['regions'] = self.visualizer.plot_influential_regions(
                spectrum, wavelengths, regions,
                save_path=save_path('influential_regions.png'),
            )
        
        # Prediction explanation
        if prediction:
            saliencies = {'overall': saliency}
            figures['explanation'] = self.visualizer.plot_prediction_explanation(
                spectrum, wavelengths, prediction, saliencies,
                save_path=save_path('prediction_explanation.png'),
            )
        
        return figures
    
    def full_analysis(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        methods: List[str] = ["integrated_gradients", "smooth_grad", "occlusion"],
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run complete explainability analysis.
        """
        results = {
            'saliencies': {},
            'regions': {},
            'figures': {},
        }
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Compute saliencies with different methods
        for method in methods:
            try:
                saliency = self.explain(
                    spectrum, wavelengths, method=method
                )
                results['saliencies'][method] = saliency
                
                regions = self.detect_regions(saliency, wavelengths)
                results['regions'][method] = regions
                
            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
        
        # Generate comparison plot
        if results['saliencies']:
            results['figures']['comparison'] = self.visualizer.plot_comparison(
                spectrum, wavelengths, results['saliencies'],
                save_path=str(Path(save_dir) / 'method_comparison.png') if save_dir else None,
            )
        
        # Best method visualization
        best_method = methods[0] if methods else 'integrated_gradients'
        if best_method in results['saliencies']:
            figs = self.visualize(
                spectrum, wavelengths,
                results['saliencies'][best_method],
                results['regions'].get(best_method),
                save_dir=save_dir,
            )
            results['figures'].update(figs)
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def explain_prediction(
    model: nn.Module,
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    method: str = "integrated_gradients",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Quick function to explain a single prediction.
    """
    explainer = SpectralExplainer(model)
    
    saliency = explainer.explain(spectrum, wavelengths, method=method)
    regions = explainer.detect_regions(saliency, wavelengths)
    
    return saliency, regions


def visualize_explanation(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    saliency: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Quick visualization function.
    """
    visualizer = ExplainabilityVisualizer()
    return visualizer.plot_saliency_map(
        spectrum, saliency, wavelengths, save_path=save_path
    )


# =============================================================================
# Main (Demo)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Explainability Tools Demo")
    print("=" * 70)
    
    # Create synthetic data
    np.random.seed(42)
    
    wavelengths = np.linspace(0.6, 5.0, 512)
    
    # Create spectrum with absorption features
    spectrum = np.ones(512)
    
    # Add some absorption lines
    features = [
        (1.4, 0.3, 0.05, "H2O"),  # center, depth, width, molecule
        (2.7, 0.25, 0.03, "CO2"),
        (3.3, 0.2, 0.04, "CH4"),
        (4.3, 0.35, 0.05, "CO2"),
    ]
    
    for center, depth, width, _ in features:
        spectrum -= depth * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
    
    # Add noise
    spectrum += 0.02 * np.random.randn(512)
    
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} μm")
    
    # Test saliency (without model - synthetic)
    print("\n--- Testing Region Detection ---")
    
    # Create synthetic saliency
    saliency = np.zeros(512)
    for center, depth, width, _ in features:
        idx = np.argmin(np.abs(wavelengths - center))
        width_idx = int(width / (wavelengths[1] - wavelengths[0]))
        start = max(0, idx - width_idx * 2)
        end = min(512, idx + width_idx * 2)
        saliency[start:end] = depth
    
    # Detect regions
    detector = InfluentialRegionDetector(threshold=0.1)
    regions = detector.detect(saliency, wavelengths)
    regions = detector.match_molecules(regions)
    
    print(f"Detected {len(regions)} influential regions:")
    for i, region in enumerate(regions[:5]):
        print(f"  Region {i+1}: {region['start_wavelength']:.2f}-{region['end_wavelength']:.2f} μm")
        print(f"    Importance: {region['max_importance']:.3f}")
        if region.get('molecule_matches'):
            matches = [f"{m['molecule']} ({m['confidence']:.2f})" 
                      for m in region['molecule_matches'][:3]]
            print(f"    Matches: {', '.join(matches)}")
    
    # Test visualization
    print("\n--- Testing Visualization ---")
    
    visualizer = ExplainabilityVisualizer()
    
    fig1 = visualizer.plot_saliency_map(
        spectrum, saliency, wavelengths,
        title="Synthetic Saliency Map",
        save_path=None,
    )
    
    fig2 = visualizer.plot_influential_regions(
        spectrum, wavelengths, regions,
        title="Detected Absorption Features",
        save_path=None,
    )
    
    # Test method comparison
    print("\n--- Testing Method Comparison ---")
    
    saliencies = {
        'Integrated Gradients': saliency,
        'SmoothGrad': saliency + 0.05 * np.random.randn(512),
        'Occlusion': np.abs(saliency) * (1 + 0.1 * np.random.randn(512)),
    }
    
    fig3 = visualizer.plot_comparison(
        spectrum, wavelengths, saliencies,
        title="Comparison of Explanation Methods",
        save_path=None,
    )
    
    print("\n--- Demo Complete ---")
    print("To use with a real model:")
    print("  explainer = SpectralExplainer(model)")
    print("  saliency = explainer.explain(spectrum, wavelengths)")
    print("  regions = explainer.detect_regions(saliency, wavelengths)")
    print("  explainer.visualize(spectrum, wavelengths, saliency, regions)")
    
    plt.show()
