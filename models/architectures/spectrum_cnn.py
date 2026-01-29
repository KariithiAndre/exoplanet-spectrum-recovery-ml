"""
1D CNN for Exoplanet Spectrum Classification

Multi-task deep learning model for:
1. Multi-label molecule detection (H2O, CO2, CH4, etc.)
2. Planet class prediction (Hot Jupiter, Warm Neptune, etc.)
3. Habitability score regression (0-1 scale)

Architecture features:
- 1D Convolutional layers with residual connections
- Batch normalization and dropout for regularization
- Multi-head output for different tasks
- Attention mechanism for feature weighting

Author: Exoplanet Spectrum Recovery Project
"""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class PlanetClass(Enum):
    """Planet classification categories."""
    HOT_JUPITER = 0
    WARM_JUPITER = 1
    COLD_JUPITER = 2
    HOT_NEPTUNE = 3
    WARM_NEPTUNE = 4
    SUPER_EARTH = 5
    EARTH_LIKE = 6
    WATER_WORLD = 7
    LAVA_WORLD = 8
    UNKNOWN = 9


class MoleculeLabel(Enum):
    """Molecules for multi-label detection."""
    H2O = 0
    CO2 = 1
    CO = 2
    CH4 = 3
    NH3 = 4
    O2 = 5
    O3 = 6
    Na = 7
    K = 8
    TiO = 9
    VO = 10
    FeH = 11
    H2S = 12
    HCN = 13


NUM_MOLECULES = len(MoleculeLabel)
NUM_PLANET_CLASSES = len(PlanetClass)
MOLECULE_NAMES = [m.name for m in MoleculeLabel]
PLANET_CLASS_NAMES = [p.name for p in PlanetClass]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the CNN model."""
    
    # Input
    input_length: int = 512  # Spectrum length
    input_channels: int = 1  # Single channel (flux only) or 2 (flux + error)
    
    # CNN Architecture
    base_channels: int = 64
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 5, 3])
    
    # Regularization
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    
    # Attention
    use_attention: bool = True
    attention_heads: int = 4
    
    # Residual connections
    use_residual: bool = True
    
    # Output heads
    num_molecules: int = NUM_MOLECULES
    num_planet_classes: int = NUM_PLANET_CLASSES
    
    # Task weights
    molecule_weight: float = 1.0
    planet_weight: float = 1.0
    habitability_weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v if not isinstance(v, list) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "onecycle", "plateau"
    warmup_epochs: int = 5
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    
    # Data
    validation_split: float = 0.2
    num_workers: int = 4
    
    # Checkpointing
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock1D(nn.Module):
    """
    1D Convolutional block with batch normalization and dropout.
    
    Conv -> BatchNorm -> ReLU -> Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock1D(nn.Module):
    """
    Residual block with two convolutions and skip connection.
    
    x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels) if use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels) if use_batch_norm else nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class SelfAttention1D(nn.Module):
    """
    Self-attention mechanism for 1D sequences.
    
    Helps the model focus on important spectral features.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, 3*C, L)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, L)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = attn @ v  # (B, heads, L, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, L)
        out = self.proj(out)
        
        return out + x  # Residual connection


class SqueezeExcitation1D(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(channels // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        # Squeeze
        y = self.squeeze(x).view(B, C)
        
        # Excitation
        y = self.excitation(y).view(B, C, 1)
        
        return x * y


# =============================================================================
# Main Model Architecture
# =============================================================================

class SpectrumEncoder(nn.Module):
    """
    1D CNN encoder for spectrum features.
    
    Progressively reduces spatial dimension while increasing channels.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        channels = config.base_channels
        in_channels = config.input_channels
        
        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 7, padding=3),
            nn.BatchNorm1d(channels) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        # Build encoder stages
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i, (mult, kernel) in enumerate(zip(
            config.channel_multipliers,
            config.kernel_sizes
        )):
            out_channels = channels * mult
            
            # Conv block
            stage = nn.Sequential(
                ConvBlock1D(
                    channels if i == 0 else channels * config.channel_multipliers[i-1],
                    out_channels,
                    kernel,
                    dropout=config.dropout_rate,
                    use_batch_norm=config.use_batch_norm,
                ),
            )
            
            # Optional residual block
            if config.use_residual:
                stage.append(ResidualBlock1D(
                    out_channels,
                    kernel,
                    dropout=config.dropout_rate,
                    use_batch_norm=config.use_batch_norm,
                ))
            
            # Optional squeeze-excitation
            stage.append(SqueezeExcitation1D(out_channels))
            
            self.stages.append(nn.Sequential(*stage))
            self.pools.append(nn.MaxPool1d(2, 2))
        
        # Final channels
        self.final_channels = channels * config.channel_multipliers[-1]
        
        # Optional attention
        if config.use_attention:
            self.attention = SelfAttention1D(
                self.final_channels,
                config.attention_heads,
                config.dropout_rate,
            )
        else:
            self.attention = nn.Identity()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, L)
            
        Returns:
            Tuple of (features, pooled_features)
        """
        x = self.stem(x)
        
        for stage, pool in zip(self.stages, self.pools):
            x = stage(x)
            x = pool(x)
        
        x = self.attention(x)
        
        # Global features
        pooled = self.global_pool(x).squeeze(-1)
        
        return x, pooled


class MoleculeHead(nn.Module):
    """
    Multi-label classification head for molecule detection.
    
    Uses sigmoid activation for independent probabilities.
    """
    
    def __init__(
        self,
        in_features: int,
        num_molecules: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_molecules),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # Raw logits, apply sigmoid in loss


class PlanetClassHead(nn.Module):
    """
    Multi-class classification head for planet type.
    
    Uses softmax activation for mutually exclusive classes.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # Raw logits, apply softmax in loss


class HabitabilityHead(nn.Module):
    """
    Regression head for habitability score (0-1).
    
    Uses sigmoid to constrain output range.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class ExoplanetSpectrumCNN(nn.Module):
    """
    Complete multi-task 1D CNN for exoplanet spectrum analysis.
    
    Tasks:
    1. Multi-label molecule detection (14 molecules)
    2. Planet class prediction (10 classes)
    3. Habitability score regression (0-1)
    
    Example:
        config = ModelConfig(input_length=512)
        model = ExoplanetSpectrumCNN(config)
        
        spectrum = torch.randn(8, 1, 512)  # Batch of 8
        outputs = model(spectrum)
        
        molecules = torch.sigmoid(outputs['molecules'])  # (8, 14)
        planet_class = torch.softmax(outputs['planet_class'], dim=-1)  # (8, 10)
        habitability = outputs['habitability']  # (8,)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # Encoder
        self.encoder = SpectrumEncoder(self.config)
        
        # Feature dimension
        feature_dim = self.encoder.final_channels
        
        # Task heads
        self.molecule_head = MoleculeHead(
            feature_dim,
            self.config.num_molecules,
            dropout=self.config.dropout_rate,
        )
        
        self.planet_head = PlanetClassHead(
            feature_dim,
            self.config.num_planet_classes,
            dropout=self.config.dropout_rate,
        )
        
        self.habitability_head = HabitabilityHead(
            feature_dim,
            dropout=self.config.dropout_rate,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input spectrum (B, C, L) or (B, L)
            return_features: If True, also return intermediate features
            
        Returns:
            Dictionary with predictions for each task
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Encode
        features, pooled = self.encoder(x)
        
        # Task predictions
        outputs = {
            'molecules': self.molecule_head(pooled),
            'planet_class': self.planet_head(pooled),
            'habitability': self.habitability_head(pooled),
        }
        
        if return_features:
            outputs['features'] = pooled
            outputs['feature_maps'] = features
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with proper activations applied.
        
        Returns probabilities/scores rather than raw logits.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            return {
                'molecules': torch.sigmoid(outputs['molecules']),
                'planet_class': F.softmax(outputs['planet_class'], dim=-1),
                'habitability': outputs['habitability'],
            }
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Dataset
# =============================================================================

class ExoplanetSpectrumDataset(Dataset):
    """
    Dataset for exoplanet spectra with multi-task labels.
    
    Expected data format:
    - spectrum: (L,) or (C, L) array
    - molecules: (num_molecules,) binary array
    - planet_class: int (0 to num_classes-1)
    - habitability: float (0 to 1)
    """
    
    def __init__(
        self,
        spectra: np.ndarray,
        molecule_labels: np.ndarray,
        planet_classes: np.ndarray,
        habitability_scores: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        augment: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            spectra: Array of spectra (N, L) or (N, C, L)
            molecule_labels: Multi-label array (N, num_molecules)
            planet_classes: Class indices (N,)
            habitability_scores: Scores (N,)
            wavelengths: Wavelength array (L,) - optional
            augment: Enable data augmentation
        """
        self.spectra = torch.from_numpy(spectra).float()
        self.molecule_labels = torch.from_numpy(molecule_labels).float()
        self.planet_classes = torch.from_numpy(planet_classes).long()
        self.habitability_scores = torch.from_numpy(habitability_scores).float()
        self.wavelengths = wavelengths
        self.augment = augment
        
        # Ensure spectra have channel dimension
        if self.spectra.dim() == 2:
            self.spectra = self.spectra.unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spectrum = self.spectra[idx]
        
        # Apply augmentation
        if self.augment:
            spectrum = self._augment(spectrum)
        
        return {
            'spectrum': spectrum,
            'molecules': self.molecule_labels[idx],
            'planet_class': self.planet_classes[idx],
            'habitability': self.habitability_scores[idx],
        }
    
    def _augment(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation."""
        # Random noise
        if torch.rand(1) < 0.5:
            noise_level = torch.rand(1) * 0.02
            spectrum = spectrum + torch.randn_like(spectrum) * noise_level
        
        # Random scaling
        if torch.rand(1) < 0.5:
            scale = 0.95 + torch.rand(1) * 0.1
            spectrum = spectrum * scale
        
        # Random shift
        if torch.rand(1) < 0.3:
            shift = (torch.rand(1) - 0.5) * 0.02
            spectrum = spectrum + shift
        
        return spectrum


# =============================================================================
# Loss Functions
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Combines:
    - Binary Cross Entropy for molecule detection
    - Cross Entropy for planet classification
    - MSE for habitability regression
    
    Supports uncertainty-based loss weighting.
    """
    
    def __init__(
        self,
        molecule_weight: float = 1.0,
        planet_weight: float = 1.0,
        habitability_weight: float = 1.0,
        use_uncertainty_weighting: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        self.molecule_weight = molecule_weight
        self.planet_weight = planet_weight
        self.habitability_weight = habitability_weight
        
        # Loss functions
        self.molecule_loss = nn.BCEWithLogitsLoss()
        self.planet_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.habitability_loss = nn.MSELoss()
        
        # Uncertainty weighting (learnable)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(3))
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate combined loss.
        
        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        # Individual losses
        mol_loss = self.molecule_loss(
            predictions['molecules'],
            targets['molecules'],
        )
        
        planet_loss = self.planet_loss(
            predictions['planet_class'],
            targets['planet_class'],
        )
        
        hab_loss = self.habitability_loss(
            predictions['habitability'],
            targets['habitability'],
        )
        
        # Combine losses
        if self.use_uncertainty_weighting:
            # Uncertainty-based weighting
            precision_mol = torch.exp(-self.log_vars[0])
            precision_planet = torch.exp(-self.log_vars[1])
            precision_hab = torch.exp(-self.log_vars[2])
            
            total_loss = (
                precision_mol * mol_loss + self.log_vars[0] +
                precision_planet * planet_loss + self.log_vars[1] +
                precision_hab * hab_loss + self.log_vars[2]
            )
        else:
            total_loss = (
                self.molecule_weight * mol_loss +
                self.planet_weight * planet_loss +
                self.habitability_weight * hab_loss
            )
        
        losses = {
            'total': total_loss,
            'molecule': mol_loss,
            'planet': planet_loss,
            'habitability': hab_loss,
        }
        
        return total_loss, losses


# =============================================================================
# Metrics
# =============================================================================

class MultiTaskMetrics:
    """
    Metrics for multi-task evaluation.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.molecule_preds = []
        self.molecule_targets = []
        self.planet_preds = []
        self.planet_targets = []
        self.hab_preds = []
        self.hab_targets = []
    
    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        """Update with batch predictions."""
        # Molecule detection
        mol_probs = torch.sigmoid(predictions['molecules'])
        self.molecule_preds.append(mol_probs.cpu())
        self.molecule_targets.append(targets['molecules'].cpu())
        
        # Planet classification
        planet_probs = F.softmax(predictions['planet_class'], dim=-1)
        self.planet_preds.append(planet_probs.cpu())
        self.planet_targets.append(targets['planet_class'].cpu())
        
        # Habitability
        self.hab_preds.append(predictions['habitability'].cpu())
        self.hab_targets.append(targets['habitability'].cpu())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        
        # Concatenate batches
        mol_preds = torch.cat(self.molecule_preds)
        mol_targets = torch.cat(self.molecule_targets)
        planet_preds = torch.cat(self.planet_preds)
        planet_targets = torch.cat(self.planet_targets)
        hab_preds = torch.cat(self.hab_preds)
        hab_targets = torch.cat(self.hab_targets)
        
        # Molecule metrics (multi-label)
        mol_binary = (mol_preds > self.threshold).float()
        
        # Per-class accuracy
        correct = (mol_binary == mol_targets).float()
        metrics['molecule_accuracy'] = correct.mean().item()
        
        # Precision, Recall, F1
        tp = (mol_binary * mol_targets).sum(dim=0)
        fp = (mol_binary * (1 - mol_targets)).sum(dim=0)
        fn = ((1 - mol_binary) * mol_targets).sum(dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics['molecule_precision'] = precision.mean().item()
        metrics['molecule_recall'] = recall.mean().item()
        metrics['molecule_f1'] = f1.mean().item()
        
        # Planet classification metrics
        planet_pred_class = planet_preds.argmax(dim=-1)
        correct = (planet_pred_class == planet_targets).float()
        metrics['planet_accuracy'] = correct.mean().item()
        
        # Top-3 accuracy
        _, top3 = planet_preds.topk(3, dim=-1)
        top3_correct = (top3 == planet_targets.unsqueeze(-1)).any(dim=-1).float()
        metrics['planet_top3_accuracy'] = top3_correct.mean().item()
        
        # Habitability metrics
        metrics['habitability_mse'] = F.mse_loss(hab_preds, hab_targets).item()
        metrics['habitability_mae'] = F.l1_loss(hab_preds, hab_targets).item()
        
        # Correlation
        if len(hab_preds) > 1:
            hab_preds_np = hab_preds.numpy()
            hab_targets_np = hab_targets.numpy()
            correlation = np.corrcoef(hab_preds_np, hab_targets_np)[0, 1]
            metrics['habitability_correlation'] = correlation if np.isfinite(correlation) else 0.0
        
        return metrics


# =============================================================================
# Training Pipeline
# =============================================================================

class Trainer:
    """
    Training pipeline for the multi-task CNN.
    
    Features:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: ExoplanetSpectrumCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Loss
        self.criterion = MultiTaskLoss(
            molecule_weight=model.config.molecule_weight,
            planet_weight=model.config.planet_weight,
            habitability_weight=model.config.habitability_weight,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.num_epochs
        
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01,
            )
        elif config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                total_steps=total_steps,
                pct_start=config.warmup_epochs / config.num_epochs,
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5,
            )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
        
        # Metrics
        self.metrics = MultiTaskMetrics()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': [],
        }
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        losses_sum = {'molecule': 0.0, 'planet': 0.0, 'habitability': 0.0}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            spectrum = batch['spectrum'].to(self.device)
            targets = {
                'molecules': batch['molecules'].to(self.device),
                'planet_class': batch['planet_class'].to(self.device),
                'habitability': batch['habitability'].to(self.device),
            }
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(spectrum)
                    loss, losses = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(spectrum)
                loss, losses = self.criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update scheduler for OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            for k, v in losses.items():
                if k != 'total':
                    losses_sum[k] += v.item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'molecule_loss': losses_sum['molecule'] / n_batches,
            'planet_loss': losses_sum['planet'] / n_batches,
            'habitability_loss': losses_sum['habitability'] / n_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        
        for batch in self.val_loader:
            spectrum = batch['spectrum'].to(self.device)
            targets = {
                'molecules': batch['molecules'].to(self.device),
                'planet_class': batch['planet_class'].to(self.device),
                'habitability': batch['habitability'].to(self.device),
            }
            
            outputs = self.model(spectrum)
            loss, _ = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            self.metrics.update(outputs, targets)
        
        val_loss = total_loss / len(self.val_loader)
        metrics = self.metrics.compute()
        
        return {'loss': val_loss}, metrics
    
    def train(self) -> Dict[str, List]:
        """
        Full training loop.
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_results = self.train_epoch()
            
            # Validate
            val_results, metrics = self.validate()
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_results['loss'])
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            
            # Log results
            logger.info(
                f"Train Loss: {train_results['loss']:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"Molecule F1: {metrics['molecule_f1']:.4f} | "
                f"Planet Acc: {metrics['planet_accuracy']:.4f} | "
                f"Hab MAE: {metrics['habitability_mae']:.4f}"
            )
            
            # Track history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['metrics'].append(metrics)
            
            # Checkpointing
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                self.epochs_without_improvement = 0
                
                if self.config.save_best:
                    self.save_checkpoint('best_model.pt')
                    logger.info("Saved best model")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if (self.config.early_stopping and 
                self.epochs_without_improvement >= self.config.patience):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']


# =============================================================================
# Convenience Functions
# =============================================================================

def create_model(
    input_length: int = 512,
    pretrained_path: Optional[str] = None,
    **config_kwargs,
) -> ExoplanetSpectrumCNN:
    """
    Create model with optional pretrained weights.
    
    Args:
        input_length: Spectrum length
        pretrained_path: Path to pretrained checkpoint
        **config_kwargs: Model configuration options
        
    Returns:
        Initialized model
    """
    config = ModelConfig(input_length=input_length, **config_kwargs)
    model = ExoplanetSpectrumCNN(config)
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
    
    return model


def create_data_loaders(
    spectra: np.ndarray,
    molecule_labels: np.ndarray,
    planet_classes: np.ndarray,
    habitability_scores: np.ndarray,
    batch_size: int = 32,
    validation_split: float = 0.2,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = ExoplanetSpectrumDataset(
        spectra, molecule_labels, planet_classes, habitability_scores,
        augment=True,
    )
    
    # Split
    n_val = int(len(dataset) * validation_split)
    n_train = len(dataset) - n_val
    
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_model(
    spectra: np.ndarray,
    molecule_labels: np.ndarray,
    planet_classes: np.ndarray,
    habitability_scores: np.ndarray,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> Tuple[ExoplanetSpectrumCNN, Dict]:
    """
    Complete training pipeline.
    
    Args:
        spectra: Input spectra (N, L)
        molecule_labels: Multi-label targets (N, num_molecules)
        planet_classes: Class indices (N,)
        habitability_scores: Habitability scores (N,)
        model_config: Model configuration
        training_config: Training configuration
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Defaults
    model_config = model_config or ModelConfig(input_length=spectra.shape[-1])
    training_config = training_config or TrainingConfig()
    
    # Create model
    model = ExoplanetSpectrumCNN(model_config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        spectra, molecule_labels, planet_classes, habitability_scores,
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        num_workers=training_config.num_workers,
    )
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, training_config)
    history = trainer.train()
    
    return model, history


# =============================================================================
# Inference Utilities
# =============================================================================

@torch.no_grad()
def predict_spectrum(
    model: ExoplanetSpectrumCNN,
    spectrum: np.ndarray,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Make predictions for a single spectrum.
    
    Args:
        model: Trained model
        spectrum: Input spectrum (L,) or (C, L)
        device: Device for inference
        
    Returns:
        Dictionary with predictions and probabilities
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Prepare input
    x = torch.from_numpy(spectrum).float()
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif x.dim() == 2:
        x = x.unsqueeze(0)  # Add batch dim
    
    x = x.to(device)
    
    # Predict
    outputs = model.predict(x)
    
    # Process results
    mol_probs = outputs['molecules'][0].cpu().numpy()
    planet_probs = outputs['planet_class'][0].cpu().numpy()
    habitability = outputs['habitability'][0].cpu().item()
    
    # Get detected molecules
    detected_molecules = [
        MOLECULE_NAMES[i] for i, p in enumerate(mol_probs) if p > 0.5
    ]
    
    # Get top planet classes
    top_classes = np.argsort(planet_probs)[::-1][:3]
    
    return {
        'detected_molecules': detected_molecules,
        'molecule_probabilities': {
            MOLECULE_NAMES[i]: float(p) for i, p in enumerate(mol_probs)
        },
        'planet_class': PLANET_CLASS_NAMES[top_classes[0]],
        'planet_probabilities': {
            PLANET_CLASS_NAMES[i]: float(planet_probs[i]) 
            for i in range(len(planet_probs))
        },
        'habitability_score': habitability,
        'top_planet_classes': [PLANET_CLASS_NAMES[i] for i in top_classes],
    }


# =============================================================================
# Main (Demo)
# =============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("Exoplanet Spectrum CNN Demo")
    print("=" * 70)
    
    # Generate synthetic data
    n_samples = 1000
    spectrum_length = 512
    
    # Random spectra
    spectra = np.random.randn(n_samples, spectrum_length).astype(np.float32)
    
    # Random labels
    molecule_labels = (np.random.rand(n_samples, NUM_MOLECULES) > 0.7).astype(np.float32)
    planet_classes = np.random.randint(0, NUM_PLANET_CLASSES, n_samples)
    habitability_scores = np.random.rand(n_samples).astype(np.float32)
    
    print(f"Data: {n_samples} samples, {spectrum_length} spectral points")
    print(f"Molecules: {NUM_MOLECULES} classes (multi-label)")
    print(f"Planet types: {NUM_PLANET_CLASSES} classes")
    
    # Create model
    config = ModelConfig(
        input_length=spectrum_length,
        base_channels=32,
        dropout_rate=0.3,
        use_attention=True,
    )
    
    model = ExoplanetSpectrumCNN(config)
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(8, 1, spectrum_length)
    outputs = model(x)
    
    print(f"\nOutput shapes:")
    print(f"  Molecules: {outputs['molecules'].shape}")
    print(f"  Planet class: {outputs['planet_class'].shape}")
    print(f"  Habitability: {outputs['habitability'].shape}")
    
    # Test prediction
    predictions = model.predict(x)
    print(f"\nPrediction shapes:")
    print(f"  Molecules: {predictions['molecules'].shape} (probabilities)")
    print(f"  Planet class: {predictions['planet_class'].shape} (probabilities)")
    print(f"  Habitability: {predictions['habitability'].shape} (scores)")
    
    # Quick training demo (few epochs)
    print("\n--- Quick Training Demo ---")
    
    train_config = TrainingConfig(
        batch_size=32,
        num_epochs=3,
        learning_rate=1e-3,
        early_stopping=False,
    )
    
    train_loader, val_loader = create_data_loaders(
        spectra, molecule_labels, planet_classes, habitability_scores,
        batch_size=train_config.batch_size,
    )
    
    trainer = Trainer(model, train_loader, val_loader, train_config)
    
    # Train for a few epochs
    for epoch in range(3):
        train_results = trainer.train_epoch()
        val_results, metrics = trainer.validate()
        print(f"Epoch {epoch + 1}: Train Loss={train_results['loss']:.4f}, "
              f"Val Loss={val_results['loss']:.4f}, "
              f"Molecule F1={metrics['molecule_f1']:.4f}")
    
    # Single prediction
    print("\n--- Single Prediction ---")
    sample_spectrum = spectra[0]
    result = predict_spectrum(model, sample_spectrum)
    
    print(f"Detected molecules: {result['detected_molecules']}")
    print(f"Planet class: {result['planet_class']}")
    print(f"Habitability: {result['habitability_score']:.3f}")
