"""
Complete ML Training Pipeline for Exoplanet Spectrum Classification

Features:
- Spectral data augmentation (noise, shifts, scaling, masking)
- Train/validation/test splitting with stratification
- Comprehensive metrics (F1, ROC-AUC, Precision, Recall)
- Model checkpoint saving with best model tracking
- Training visualization and logging

Author: Exoplanet Spectrum Recovery Project
"""

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

MOLECULE_NAMES = [
    "H2O", "CO2", "CO", "CH4", "NH3", "O2", "O3",
    "Na", "K", "TiO", "VO", "FeH", "H2S", "HCN"
]
PLANET_CLASS_NAMES = [
    "HOT_JUPITER", "WARM_JUPITER", "COLD_JUPITER", "HOT_NEPTUNE",
    "WARM_NEPTUNE", "SUPER_EARTH", "EARTH_LIKE", "WATER_WORLD",
    "LAVA_WORLD", "UNKNOWN"
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Noise augmentation
    add_noise: bool = True
    noise_std_range: Tuple[float, float] = (0.01, 0.1)
    
    # Wavelength shift (simulates calibration errors)
    wavelength_shift: bool = True
    shift_range: Tuple[float, float] = (-0.02, 0.02)  # Fraction
    
    # Intensity scaling
    intensity_scale: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)
    
    # Random masking (simulates bad pixels)
    random_mask: bool = True
    mask_prob: float = 0.05
    mask_value: float = 0.0
    
    # Continuum variation
    continuum_variation: bool = True
    poly_degree: int = 2
    poly_amplitude: float = 0.1
    
    # Random crop and pad
    random_crop: bool = False
    crop_fraction: float = 0.9
    
    # Mixup augmentation
    mixup: bool = False
    mixup_alpha: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model
    model_type: str = "transformer"  # "cnn" or "transformer"
    input_length: int = 512
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    save_best_only: bool = True
    
    # Logging
    log_dir: str = "logs"
    log_every: int = 10
    
    # Device
    device: str = "auto"
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Augmentation
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


# =============================================================================
# Data Augmentation
# =============================================================================

class SpectralAugmentation:
    """
    Data augmentation techniques for spectral data.
    
    All augmentations are designed to preserve the physical
    characteristics of transmission spectra while adding variability.
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def __call__(
        self,
        spectrum: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random augmentations."""
        spectrum = spectrum.copy()
        if wavelengths is not None:
            wavelengths = wavelengths.copy()
        
        # Apply augmentations with probability
        if self.config.add_noise and random.random() < 0.5:
            spectrum = self._add_noise(spectrum)
        
        if self.config.wavelength_shift and random.random() < 0.3:
            spectrum, wavelengths = self._wavelength_shift(spectrum, wavelengths)
        
        if self.config.intensity_scale and random.random() < 0.5:
            spectrum = self._intensity_scale(spectrum)
        
        if self.config.random_mask and random.random() < 0.3:
            spectrum = self._random_mask(spectrum)
        
        if self.config.continuum_variation and random.random() < 0.4:
            spectrum = self._continuum_variation(spectrum)
        
        if self.config.random_crop and random.random() < 0.2:
            spectrum, wavelengths = self._random_crop(spectrum, wavelengths)
        
        return spectrum, wavelengths
    
    def _add_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        std = np.random.uniform(*self.config.noise_std_range)
        noise = np.random.normal(0, std, spectrum.shape)
        return spectrum + noise
    
    def _wavelength_shift(
        self,
        spectrum: np.ndarray,
        wavelengths: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply wavelength shift via interpolation."""
        shift = np.random.uniform(*self.config.shift_range)
        n = len(spectrum)
        
        # Shift indices
        old_idx = np.arange(n)
        new_idx = old_idx + shift * n
        
        # Interpolate
        spectrum = np.interp(old_idx, new_idx, spectrum)
        
        return spectrum, wavelengths
    
    def _intensity_scale(self, spectrum: np.ndarray) -> np.ndarray:
        """Scale intensity values."""
        scale = np.random.uniform(*self.config.scale_range)
        return spectrum * scale
    
    def _random_mask(self, spectrum: np.ndarray) -> np.ndarray:
        """Randomly mask some values (simulates bad pixels)."""
        mask = np.random.random(spectrum.shape) < self.config.mask_prob
        spectrum[mask] = self.config.mask_value
        return spectrum
    
    def _continuum_variation(self, spectrum: np.ndarray) -> np.ndarray:
        """Add polynomial continuum variation."""
        n = len(spectrum)
        x = np.linspace(-1, 1, n)
        
        # Random polynomial coefficients
        coeffs = np.random.uniform(
            -self.config.poly_amplitude,
            self.config.poly_amplitude,
            self.config.poly_degree + 1
        )
        
        continuum = np.polyval(coeffs, x)
        return spectrum + continuum
    
    def _random_crop(
        self,
        spectrum: np.ndarray,
        wavelengths: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Random crop and resize."""
        n = len(spectrum)
        crop_size = int(n * self.config.crop_fraction)
        
        start = np.random.randint(0, n - crop_size)
        end = start + crop_size
        
        cropped = spectrum[start:end]
        
        # Resize back to original length
        x_old = np.linspace(0, 1, crop_size)
        x_new = np.linspace(0, 1, n)
        spectrum = np.interp(x_new, x_old, cropped)
        
        if wavelengths is not None:
            wavelengths = np.interp(x_new, x_old, wavelengths[start:end])
        
        return spectrum, wavelengths
    
    def mixup(
        self,
        spectrum1: np.ndarray,
        spectrum2: np.ndarray,
        labels1: Dict[str, np.ndarray],
        labels2: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Mixup augmentation."""
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        spectrum = lam * spectrum1 + (1 - lam) * spectrum2
        
        labels = {}
        for key in labels1:
            labels[key] = lam * labels1[key] + (1 - lam) * labels2[key]
        
        return spectrum, labels


# =============================================================================
# Dataset
# =============================================================================

class ExoplanetSpectrumDataset(Dataset):
    """
    Dataset for exoplanet spectrum classification.
    """
    
    def __init__(
        self,
        spectra: np.ndarray,
        wavelengths: Optional[np.ndarray],
        molecules: np.ndarray,
        planet_classes: np.ndarray,
        habitability: np.ndarray,
        augmentation: Optional[SpectralAugmentation] = None,
        training: bool = True,
    ):
        """
        Args:
            spectra: (N, length) array of spectra
            wavelengths: (length,) or (N, length) array of wavelengths
            molecules: (N, num_molecules) binary array
            planet_classes: (N,) class indices
            habitability: (N,) habitability scores
            augmentation: Optional augmentation
            training: Whether in training mode
        """
        self.spectra = spectra.astype(np.float32)
        self.wavelengths = wavelengths.astype(np.float32) if wavelengths is not None else None
        self.molecules = molecules.astype(np.float32)
        self.planet_classes = planet_classes.astype(np.int64)
        self.habitability = habitability.astype(np.float32)
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spectrum = self.spectra[idx]
        
        if self.wavelengths is not None:
            if self.wavelengths.ndim == 1:
                wavelengths = self.wavelengths
            else:
                wavelengths = self.wavelengths[idx]
        else:
            wavelengths = None
        
        # Apply augmentation during training
        if self.training and self.augmentation is not None:
            spectrum, wavelengths = self.augmentation(spectrum, wavelengths)
        
        result = {
            'spectrum': torch.from_numpy(spectrum),
            'molecules': torch.from_numpy(self.molecules[idx]),
            'planet_class': torch.tensor(self.planet_classes[idx]),
            'habitability': torch.tensor(self.habitability[idx]),
        }
        
        if wavelengths is not None:
            result['wavelengths'] = torch.from_numpy(wavelengths)
        
        return result


# =============================================================================
# Data Splitting
# =============================================================================

def split_data(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_labels: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        stratify_labels: Labels for stratified splitting
        seed: Random seed
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(dataset)
    indices = np.arange(n)
    
    # First split: train+val vs test
    train_val_ratio = train_ratio + val_ratio
    
    if stratify_labels is not None:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            stratify=stratify_labels,
            random_state=seed,
        )
        # Second split: train vs val
        val_size = val_ratio / train_val_ratio
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=stratify_labels[train_val_idx],
            random_state=seed,
        )
    else:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
        )
        val_size = val_ratio / train_val_ratio
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=seed,
        )
    
    logger.info(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    class_weights: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders with optional class balancing.
    """
    # Optional weighted sampling for imbalanced classes
    if class_weights is not None:
        # Get class indices from dataset
        if hasattr(train_dataset, 'dataset'):
            # It's a Subset
            indices = train_dataset.indices
            classes = train_dataset.dataset.planet_classes[indices]
        else:
            classes = train_dataset.planet_classes
        
        sample_weights = class_weights[classes]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Metrics
# =============================================================================

class MetricsCalculator:
    """
    Calculate comprehensive classification and regression metrics.
    """
    
    def __init__(self, num_molecules: int = 14, num_classes: int = 10):
        self.num_molecules = num_molecules
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.mol_preds = []
        self.mol_targets = []
        self.planet_preds = []
        self.planet_probs = []
        self.planet_targets = []
        self.hab_preds = []
        self.hab_targets = []
    
    def update(
        self,
        mol_pred: torch.Tensor,
        mol_target: torch.Tensor,
        planet_pred: torch.Tensor,
        planet_target: torch.Tensor,
        hab_pred: torch.Tensor,
        hab_target: torch.Tensor,
    ):
        """Add batch predictions."""
        self.mol_preds.append(torch.sigmoid(mol_pred).cpu().numpy())
        self.mol_targets.append(mol_target.cpu().numpy())
        
        planet_probs = F.softmax(planet_pred, dim=-1)
        self.planet_probs.append(planet_probs.cpu().numpy())
        self.planet_preds.append(planet_pred.argmax(dim=-1).cpu().numpy())
        self.planet_targets.append(planet_target.cpu().numpy())
        
        self.hab_preds.append(hab_pred.cpu().numpy())
        self.hab_targets.append(hab_target.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        
        # Concatenate predictions
        mol_preds = np.concatenate(self.mol_preds)
        mol_targets = np.concatenate(self.mol_targets)
        planet_probs = np.concatenate(self.planet_probs)
        planet_preds = np.concatenate(self.planet_preds)
        planet_targets = np.concatenate(self.planet_targets)
        hab_preds = np.concatenate(self.hab_preds)
        hab_targets = np.concatenate(self.hab_targets)
        
        # Molecule detection metrics (multi-label)
        mol_binary = (mol_preds > 0.5).astype(int)
        
        metrics['molecule_f1_micro'] = f1_score(
            mol_targets, mol_binary, average='micro', zero_division=0
        )
        metrics['molecule_f1_macro'] = f1_score(
            mol_targets, mol_binary, average='macro', zero_division=0
        )
        metrics['molecule_precision'] = precision_score(
            mol_targets, mol_binary, average='micro', zero_division=0
        )
        metrics['molecule_recall'] = recall_score(
            mol_targets, mol_binary, average='micro', zero_division=0
        )
        
        # Per-molecule ROC-AUC
        mol_aucs = []
        for i in range(self.num_molecules):
            if mol_targets[:, i].sum() > 0 and mol_targets[:, i].sum() < len(mol_targets):
                try:
                    auc = roc_auc_score(mol_targets[:, i], mol_preds[:, i])
                    mol_aucs.append(auc)
                except:
                    pass
        
        if mol_aucs:
            metrics['molecule_roc_auc'] = np.mean(mol_aucs)
        
        # Planet classification metrics
        metrics['planet_accuracy'] = accuracy_score(planet_targets, planet_preds)
        metrics['planet_f1_weighted'] = f1_score(
            planet_targets, planet_preds, average='weighted', zero_division=0
        )
        metrics['planet_f1_macro'] = f1_score(
            planet_targets, planet_preds, average='macro', zero_division=0
        )
        metrics['planet_precision'] = precision_score(
            planet_targets, planet_preds, average='weighted', zero_division=0
        )
        metrics['planet_recall'] = recall_score(
            planet_targets, planet_preds, average='weighted', zero_division=0
        )
        
        # Multi-class ROC-AUC
        try:
            # One-hot encode targets
            planet_targets_onehot = np.eye(self.num_classes)[planet_targets]
            metrics['planet_roc_auc'] = roc_auc_score(
                planet_targets_onehot, planet_probs, average='weighted', multi_class='ovr'
            )
        except:
            pass
        
        # Habitability regression metrics
        hab_error = hab_preds - hab_targets
        metrics['habitability_mae'] = np.abs(hab_error).mean()
        metrics['habitability_mse'] = (hab_error ** 2).mean()
        metrics['habitability_rmse'] = np.sqrt(metrics['habitability_mse'])
        
        # R² score
        ss_res = np.sum(hab_error ** 2)
        ss_tot = np.sum((hab_targets - hab_targets.mean()) ** 2)
        metrics['habitability_r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        
        return metrics
    
    def get_detailed_report(self) -> str:
        """Generate detailed classification report."""
        planet_preds = np.concatenate(self.planet_preds)
        planet_targets = np.concatenate(self.planet_targets)
        
        report = classification_report(
            planet_targets,
            planet_preds,
            target_names=PLANET_CLASS_NAMES,
            zero_division=0,
        )
        
        return report
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for planet classification."""
        planet_preds = np.concatenate(self.planet_preds)
        planet_targets = np.concatenate(self.planet_targets)
        
        return confusion_matrix(planet_targets, planet_preds)


# =============================================================================
# Checkpointing
# =============================================================================

class CheckpointManager:
    """
    Manage model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = True,
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        
        self.best_metric = float('-inf')
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        metric_name: str = 'val_loss',
        minimize: bool = True,
    ) -> Optional[str]:
        """
        Save checkpoint if conditions are met.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Current metrics
            metric_name: Metric to track for best model
            minimize: Whether to minimize (loss) or maximize (accuracy)
            
        Returns:
            Path to saved checkpoint or None
        """
        current_metric = metrics.get(metric_name, 0)
        
        # Check if this is the best model
        is_best = False
        if minimize:
            if current_metric < -self.best_metric:
                self.best_metric = -current_metric
                is_best = True
        else:
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                is_best = True
        
        if self.save_best_only and not is_best:
            return None
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filepath = self.checkpoint_dir / "best_model.pt"
        else:
            filepath = self.checkpoint_dir / f"checkpoint_epoch{epoch}_{timestamp}.pt"
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
        # Track checkpoints
        if not is_best:
            self.checkpoints.append(filepath)
            
            # Remove old checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
        
        return str(filepath)
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint


# =============================================================================
# Training Visualization
# =============================================================================

class TrainingVisualizer:
    """
    Visualize training progress and results.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        self.metrics_history = {}
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        metrics: Dict[str, float],
    ):
        """Update history with new epoch data."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(learning_rate)
        
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def plot_losses(self, save: bool = True) -> plt.Figure:
        """Plot training and validation losses."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.log_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_metrics(self, save: bool = True) -> plt.Figure:
        """Plot all tracked metrics."""
        n_metrics = len(self.metrics_history)
        if n_metrics == 0:
            return None
        
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        
        epochs = range(1, len(list(self.metrics_history.values())[0]) + 1)
        
        for i, (name, values) in enumerate(self.metrics_history.items()):
            ax = axes[i]
            ax.plot(epochs, values, 'b-', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(name.replace('_', ' ').title(), fontsize=10)
            ax.set_title(name.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.log_dir / 'metrics.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save: bool = True,
    ) -> plt.Figure:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True Label',
            xlabel='Predicted Label',
            title='Confusion Matrix (Normalized)',
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add text annotations
        thresh = cm_norm.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black',
                    fontsize=8,
                )
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.log_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        class_names: List[str],
        save: bool = True,
    ) -> plt.Figure:
        """Plot ROC curves for each class."""
        from sklearn.metrics import roc_curve, auc
        
        n_classes = len(class_names)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # One-hot encode if needed
        if y_true.ndim == 1:
            y_true_onehot = np.eye(n_classes)[y_true.astype(int)]
        else:
            y_true_onehot = y_true
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, (name, color) in enumerate(zip(class_names, colors)):
            if y_true_onehot[:, i].sum() == 0:
                continue
                
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.log_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_history(self):
        """Save training history to JSON."""
        history = {
            'losses': self.history,
            'metrics': self.metrics_history,
        }
        
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """
    Complete training pipeline for spectral classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.molecule_loss = nn.BCEWithLogitsLoss()
        self.planet_loss = nn.CrossEntropyLoss()
        self.habitability_loss = nn.MSELoss()
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
        
        # Metrics
        self.metrics_calc = MetricsCalculator()
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            save_best_only=config.save_best_only,
        )
        
        # Visualization
        self.visualizer = TrainingVisualizer(config.log_dir)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        if self.config.scheduler == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        
        return None
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined loss."""
        mol_loss = self.molecule_loss(outputs['molecules'], targets['molecules'])
        planet_loss = self.planet_loss(outputs['planet_class'], targets['planet_class'])
        hab_loss = self.habitability_loss(outputs['habitability'], targets['habitability'])
        
        # Weighted combination
        total_loss = mol_loss + planet_loss + 0.5 * hab_loss
        
        return total_loss
    
    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            spectrum = batch['spectrum'].to(self.device)
            wavelengths = batch.get('wavelengths')
            if wavelengths is not None:
                wavelengths = wavelengths.to(self.device)
            
            targets = {
                'molecules': batch['molecules'].to(self.device),
                'planet_class': batch['planet_class'].to(self.device),
                'habitability': batch['habitability'].to(self.device),
            }
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(spectrum, wavelengths)
                    loss = self._compute_loss(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(spectrum, wavelengths)
                loss = self._compute_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            if self.scheduler is not None and self.config.scheduler != "plateau":
                self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Tuple[float, Dict[str, float]]:
        """Run validation."""
        self.model.eval()
        self.metrics_calc.reset()
        
        loader = loader or self.val_loader
        total_loss = 0.0
        
        for batch in loader:
            spectrum = batch['spectrum'].to(self.device)
            wavelengths = batch.get('wavelengths')
            if wavelengths is not None:
                wavelengths = wavelengths.to(self.device)
            
            targets = {
                'molecules': batch['molecules'].to(self.device),
                'planet_class': batch['planet_class'].to(self.device),
                'habitability': batch['habitability'].to(self.device),
            }
            
            outputs = self.model(spectrum, wavelengths)
            loss = self._compute_loss(outputs, targets)
            
            total_loss += loss.item()
            
            # Update metrics
            self.metrics_calc.update(
                outputs['molecules'],
                targets['molecules'],
                outputs['planet_class'],
                targets['planet_class'],
                outputs['habitability'],
                targets['habitability'],
            )
        
        val_loss = total_loss / len(loader)
        metrics = self.metrics_calc.compute()
        
        return val_loss, metrics
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info("=" * 70)
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, metrics = self.validate()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if self.config.scheduler == "plateau" and self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Update visualizer
            self.visualizer.update(epoch, train_loss, val_loss, current_lr, metrics)
            
            # Logging
            if epoch % self.config.log_every == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"F1: {metrics.get('molecule_f1_micro', 0):.4f} | "
                    f"Acc: {metrics.get('planet_accuracy', 0):.4f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Checkpointing
            metrics['val_loss'] = val_loss
            self.checkpoint_manager.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                metrics,
                metric_name='val_loss',
                minimize=True,
            )
            
            # Early stopping
            if self.config.early_stopping:
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        # Final evaluation
        logger.info("\n" + "=" * 70)
        logger.info("Training Complete - Running Final Evaluation")
        logger.info("=" * 70)
        
        # Load best model
        self.checkpoint_manager.load(self.model)
        
        # Test evaluation
        if self.test_loader is not None:
            test_loss, test_metrics = self.validate(self.test_loader)
            logger.info("\nTest Results:")
            for key, value in sorted(test_metrics.items()):
                logger.info(f"  {key}: {value:.4f}")
        
        # Generate visualizations
        self.visualizer.plot_losses()
        self.visualizer.plot_metrics()
        
        # Confusion matrix
        cm = self.metrics_calc.get_confusion_matrix()
        self.visualizer.plot_confusion_matrix(cm, PLANET_CLASS_NAMES)
        
        # Save history
        self.visualizer.save_history()
        
        # Classification report
        report = self.metrics_calc.get_detailed_report()
        logger.info("\nClassification Report:")
        logger.info(report)
        
        return {
            'history': self.visualizer.history,
            'metrics': self.visualizer.metrics_history,
            'test_metrics': test_metrics if self.test_loader else None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_synthetic_data(
    n_samples: int = 1000,
    seq_length: int = 512,
    n_molecules: int = 14,
    n_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic spectral data for testing.
    """
    # Generate spectra with some structure
    wavelengths = np.linspace(0.6, 5.0, seq_length)
    spectra = np.zeros((n_samples, seq_length))
    
    for i in range(n_samples):
        # Base continuum
        spectra[i] = 1.0 + 0.1 * np.random.randn(seq_length)
        
        # Add some absorption features
        n_features = np.random.randint(3, 10)
        for _ in range(n_features):
            center = np.random.uniform(0.8, 4.5)
            depth = np.random.uniform(0.05, 0.3)
            width = np.random.uniform(0.01, 0.1)
            spectra[i] -= depth * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
    
    # Labels
    molecules = (np.random.random((n_samples, n_molecules)) > 0.7).astype(float)
    planet_classes = np.random.randint(0, n_classes, n_samples)
    habitability = np.random.random(n_samples)
    
    return spectra, wavelengths, molecules, planet_classes, habitability


def train_model(
    model: nn.Module,
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    molecules: np.ndarray,
    planet_classes: np.ndarray,
    habitability: np.ndarray,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """
    High-level function to train a model.
    """
    config = config or TrainingConfig()
    set_seed(config.seed)
    
    # Create dataset
    augmentation = SpectralAugmentation(config.augmentation)
    
    full_dataset = ExoplanetSpectrumDataset(
        spectra, wavelengths, molecules, planet_classes, habitability,
        augmentation=augmentation,
        training=True,
    )
    
    # Split data
    train_dataset, val_dataset, test_dataset = split_data(
        full_dataset,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        stratify_labels=planet_classes if config.stratify else None,
        seed=config.seed,
    )
    
    # Disable augmentation for val/test
    val_dataset.dataset.training = False
    test_dataset.dataset.training = False
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size,
    )
    
    # Train
    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    results = trainer.train()
    
    return results


# =============================================================================
# Main (Demo)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML Training Pipeline Demo")
    print("=" * 70)
    
    set_seed(42)
    
    # Generate synthetic data
    print("\n--- Generating Synthetic Data ---")
    spectra, wavelengths, molecules, planet_classes, habitability = generate_synthetic_data(
        n_samples=500,
        seq_length=512,
    )
    print(f"Spectra shape: {spectra.shape}")
    print(f"Wavelengths shape: {wavelengths.shape}")
    print(f"Molecules shape: {molecules.shape}")
    print(f"Planet classes shape: {planet_classes.shape}")
    
    # Test augmentation
    print("\n--- Testing Augmentation ---")
    aug_config = AugmentationConfig()
    augmentation = SpectralAugmentation(aug_config)
    
    orig_spectrum = spectra[0].copy()
    aug_spectrum, _ = augmentation(orig_spectrum, wavelengths)
    
    print(f"Original mean: {orig_spectrum.mean():.4f}")
    print(f"Augmented mean: {aug_spectrum.mean():.4f}")
    
    # Test dataset and splitting
    print("\n--- Testing Dataset and Splitting ---")
    dataset = ExoplanetSpectrumDataset(
        spectra, wavelengths, molecules, planet_classes, habitability,
        augmentation=augmentation,
    )
    
    train_set, val_set, test_set = split_data(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_labels=planet_classes,
    )
    
    # Test data loader
    train_loader, val_loader, test_loader = create_data_loaders(
        train_set, val_set, test_set, batch_size=16
    )
    
    batch = next(iter(train_loader))
    print(f"Batch spectrum shape: {batch['spectrum'].shape}")
    print(f"Batch molecules shape: {batch['molecules'].shape}")
    
    # Test metrics
    print("\n--- Testing Metrics ---")
    metrics_calc = MetricsCalculator()
    
    # Simulate predictions
    mol_pred = torch.randn(32, 14)
    mol_target = (torch.rand(32, 14) > 0.5).float()
    planet_pred = torch.randn(32, 10)
    planet_target = torch.randint(0, 10, (32,))
    hab_pred = torch.sigmoid(torch.randn(32))
    hab_target = torch.rand(32)
    
    metrics_calc.update(mol_pred, mol_target, planet_pred, planet_target, hab_pred, hab_target)
    metrics = metrics_calc.compute()
    
    print("Computed metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
    
    # Test visualization
    print("\n--- Testing Visualization ---")
    visualizer = TrainingVisualizer("logs/test_run")
    
    for epoch in range(10):
        visualizer.update(
            epoch + 1,
            train_loss=1.0 - 0.05 * epoch + 0.1 * np.random.random(),
            val_loss=1.0 - 0.04 * epoch + 0.15 * np.random.random(),
            learning_rate=1e-4 * (0.9 ** epoch),
            metrics={
                'accuracy': 0.5 + 0.04 * epoch,
                'f1_score': 0.4 + 0.05 * epoch,
            }
        )
    
    visualizer.plot_losses(save=False)
    visualizer.plot_metrics(save=False)
    
    print("\n--- Demo Complete ---")
    print("To run full training, use the train_model() function with real data.")
    
    plt.show()
