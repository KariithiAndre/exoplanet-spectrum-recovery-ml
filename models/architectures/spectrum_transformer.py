"""
Transformer Model for Spectral Sequence Classification

Transformer-based architecture optimized for wavelength-series learning in
exoplanet transmission spectra analysis.

Features:
1. Multiple positional encoding schemes (sinusoidal, learnable, wavelength-aware)
2. Transformer encoder with self-attention
3. Multi-task classification and regression heads
4. Uncertainty estimation (MC Dropout, Deep Ensembles, Evidential)
5. Spectral-specific optimizations

Author: Exoplanet Spectrum Recovery Project
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Molecule and planet class definitions (shared with CNN model)
MOLECULE_NAMES = [
    "H2O", "CO2", "CO", "CH4", "NH3", "O2", "O3", 
    "Na", "K", "TiO", "VO", "FeH", "H2S", "HCN"
]
PLANET_CLASS_NAMES = [
    "HOT_JUPITER", "WARM_JUPITER", "COLD_JUPITER", "HOT_NEPTUNE",
    "WARM_NEPTUNE", "SUPER_EARTH", "EARTH_LIKE", "WATER_WORLD",
    "LAVA_WORLD", "UNKNOWN"
]

NUM_MOLECULES = len(MOLECULE_NAMES)
NUM_PLANET_CLASSES = len(PLANET_CLASS_NAMES)


# =============================================================================
# Configuration
# =============================================================================

class PositionalEncodingType(Enum):
    """Types of positional encoding."""
    SINUSOIDAL = "sinusoidal"
    LEARNABLE = "learnable"
    WAVELENGTH_AWARE = "wavelength_aware"
    ROTARY = "rotary"
    ALIBI = "alibi"


class UncertaintyMethod(Enum):
    """Uncertainty estimation methods."""
    NONE = "none"
    MC_DROPOUT = "mc_dropout"
    DEEP_ENSEMBLE = "deep_ensemble"
    EVIDENTIAL = "evidential"
    HETEROSCEDASTIC = "heteroscedastic"


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    
    # Input
    input_length: int = 512
    input_channels: int = 1
    
    # Embedding
    d_model: int = 256
    positional_encoding: PositionalEncodingType = PositionalEncodingType.WAVELENGTH_AWARE
    max_wavelength: float = 30.0  # For wavelength-aware encoding
    
    # Transformer
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024  # Feed-forward dimension
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Pooling
    pooling: str = "cls"  # "cls", "mean", "max", "attention"
    
    # Output
    num_molecules: int = NUM_MOLECULES
    num_planet_classes: int = NUM_PLANET_CLASSES
    
    # Uncertainty
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.MC_DROPOUT
    mc_samples: int = 20
    
    # Regularization
    layer_norm_eps: float = 1e-6
    use_pre_norm: bool = True  # Pre-LayerNorm (more stable)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }


# =============================================================================
# Positional Encodings
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings.
    
    More flexible than sinusoidal but requires more parameters.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
        # Initialize with sinusoidal pattern
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        init_pe = torch.zeros(max_len, d_model)
        init_pe[:, 0::2] = torch.sin(position * div_term)
        init_pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe.weight.data.copy_(init_pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pe(positions)
        return self.dropout(x)


class WavelengthAwarePositionalEncoding(nn.Module):
    """
    Wavelength-aware positional encoding for spectral data.
    
    Uses actual wavelength values instead of integer positions,
    allowing the model to understand physical wavelength relationships.
    
    Encodes wavelength using logarithmic scaling (natural for spectra)
    combined with sinusoidal functions.
    """
    
    def __init__(
        self,
        d_model: int,
        max_wavelength: float = 30.0,
        min_wavelength: float = 0.1,
        dropout: float = 0.1,
        use_log_scale: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_wavelength = max_wavelength
        self.min_wavelength = min_wavelength
        self.use_log_scale = use_log_scale
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable wavelength projection
        self.wavelength_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Frequency bases for multi-scale encoding
        # Different frequencies capture features at different scales
        n_frequencies = d_model // 4
        frequencies = torch.exp(
            torch.linspace(
                math.log(0.1),  # Low frequency for broad features
                math.log(100.0),  # High frequency for narrow lines
                n_frequencies
            )
        )
        self.register_buffer('frequencies', frequencies)
    
    def forward(
        self,
        x: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            wavelengths: (batch, seq_len) or (seq_len,) - actual wavelength values
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if wavelengths is None:
            # Default: linear spacing from min to max wavelength
            wavelengths = torch.linspace(
                self.min_wavelength,
                self.max_wavelength,
                seq_len,
                device=device
            )
        
        if wavelengths.dim() == 1:
            wavelengths = wavelengths.unsqueeze(0).expand(batch_size, -1)
        
        # Normalize wavelengths
        if self.use_log_scale:
            # Log scale is natural for spectra
            wl_normalized = (
                torch.log(wavelengths + 1e-6) - math.log(self.min_wavelength)
            ) / (math.log(self.max_wavelength) - math.log(self.min_wavelength))
        else:
            wl_normalized = (wavelengths - self.min_wavelength) / (
                self.max_wavelength - self.min_wavelength
            )
        
        # Multi-frequency sinusoidal encoding
        wl_expanded = wl_normalized.unsqueeze(-1)  # (batch, seq, 1)
        
        # Apply different frequencies
        freq_encoding = wl_expanded * self.frequencies  # (batch, seq, n_freq)
        
        sin_enc = torch.sin(2 * math.pi * freq_encoding)
        cos_enc = torch.cos(2 * math.pi * freq_encoding)
        
        # Concatenate sin and cos
        freq_pe = torch.cat([sin_enc, cos_enc], dim=-1)  # (batch, seq, d_model//2)
        
        # Learnable projection from raw wavelength
        learned_pe = self.wavelength_proj(wl_expanded)  # (batch, seq, d_model)
        
        # Combine frequency encoding and learned projection
        # Use residual-style combination
        pe = learned_pe + F.pad(freq_pe, (0, self.d_model - freq_pe.size(-1)))
        
        x = x + pe
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer.
    
    Applies rotation to query and key vectors based on position.
    More effective for capturing relative positions.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Compute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache cos and sin
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        
        # Cache complex exponential
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary encoding to input."""
        seq_len = x.size(1)
        
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary transformation."""
        # Split into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)
        
        return rotated.flatten(-2)


# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional modifications for spectral data.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rotary: bool = False,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary = RotaryPositionalEncoding(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: If True, also return attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary encoding if enabled
        if self.use_rotary:
            q = self.rotary(q)
            k = self.rotary(k)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ v
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with Pre-LN or Post-LN.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_pre_norm: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.use_pre_norm = use_pre_norm
        
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, attention_dropout
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_pre_norm:
            # Pre-LayerNorm (more stable training)
            x = x + self.dropout1(self.self_attn(self.norm1(x), mask))
            x = x + self.dropout2(self.ff(self.norm2(x)))
        else:
            # Post-LayerNorm (original Transformer)
            x = self.norm1(x + self.dropout1(self.self_attn(x, mask)))
            x = self.norm2(x + self.dropout2(self.ff(x)))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers.
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_pre_norm: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, attention_dropout,
                use_pre_norm, layer_norm_eps
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm for pre-norm architecture
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if use_pre_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


# =============================================================================
# Pooling Strategies
# =============================================================================

class CLSPooling(nn.Module):
    """Use [CLS] token for sequence representation."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def add_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Add CLS token to sequence."""
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, x], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token representation."""
        return x[:, 0]


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, d_model)
        """
        # Compute attention weights
        weights = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)
        
        return pooled


# =============================================================================
# Uncertainty Estimation Modules
# =============================================================================

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Keeps dropout active during inference and runs multiple forward passes.
    """
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always apply dropout (even in eval mode)
        return F.dropout(x, self.p, training=True)


class EvidentialHead(nn.Module):
    """
    Evidential deep learning head for uncertainty estimation.
    
    For regression: Outputs parameters of Normal-Inverse-Gamma distribution.
    For classification: Outputs Dirichlet concentration parameters.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        task: str = "regression",  # "regression" or "classification"
    ):
        super().__init__()
        self.task = task
        self.out_features = out_features
        
        if task == "regression":
            # Output: (gamma, nu, alpha, beta) for each output
            self.head = nn.Linear(in_features, out_features * 4)
        else:
            # Output: Dirichlet concentration parameters
            self.head = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.head(x)
        
        if self.task == "regression":
            # Split into NIG parameters
            out = out.view(-1, self.out_features, 4)
            
            gamma = out[..., 0]  # Mean
            nu = F.softplus(out[..., 1]) + 1e-6  # Virtual observations (>0)
            alpha = F.softplus(out[..., 2]) + 1  # Shape (>1)
            beta = F.softplus(out[..., 3]) + 1e-6  # Scale (>0)
            
            # Compute uncertainty
            aleatoric = beta / (alpha - 1)
            epistemic = beta / (nu * (alpha - 1))
            
            return {
                'mean': gamma,
                'aleatoric_uncertainty': aleatoric,
                'epistemic_uncertainty': epistemic,
                'total_uncertainty': aleatoric + epistemic,
                'nig_params': (gamma, nu, alpha, beta),
            }
        else:
            # Dirichlet parameters
            alpha = F.softplus(out) + 1  # Concentration > 1
            
            # Compute probabilities and uncertainty
            S = alpha.sum(dim=-1, keepdim=True)
            probs = alpha / S
            
            # Uncertainty as total evidence
            uncertainty = self.out_features / S.squeeze(-1)
            
            return {
                'probabilities': probs,
                'alpha': alpha,
                'uncertainty': uncertainty,
            }


class HeteroscedasticHead(nn.Module):
    """
    Heteroscedastic regression head that predicts both mean and variance.
    """
    
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        
        self.mean_head = nn.Linear(in_features, out_features)
        self.var_head = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = self.mean_head(x)
        log_var = self.var_head(x)
        var = F.softplus(log_var) + 1e-6
        
        return {
            'mean': mean.squeeze(-1),
            'variance': var.squeeze(-1),
            'std': torch.sqrt(var).squeeze(-1),
        }


# =============================================================================
# Main Model
# =============================================================================

class SpectralTransformer(nn.Module):
    """
    Transformer model for spectral sequence classification.
    
    Optimized for wavelength-series learning with:
    - Wavelength-aware positional encoding
    - Multi-head self-attention
    - Multi-task outputs (molecules, planet class, habitability)
    - Uncertainty estimation
    
    Example:
        config = TransformerConfig(input_length=512)
        model = SpectralTransformer(config)
        
        spectrum = torch.randn(8, 512)
        wavelengths = torch.linspace(0.6, 5.0, 512)
        
        outputs = model(spectrum, wavelengths)
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        
        self.config = config or TransformerConfig()
        
        # Input embedding
        self.input_proj = nn.Linear(self.config.input_channels, self.config.d_model)
        
        # Positional encoding
        self._build_positional_encoding()
        
        # CLS token for pooling
        if self.config.pooling == "cls":
            self.cls_pooling = CLSPooling(self.config.d_model)
        elif self.config.pooling == "attention":
            self.attention_pooling = AttentionPooling(self.config.d_model)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            use_pre_norm=self.config.use_pre_norm,
            layer_norm_eps=self.config.layer_norm_eps,
        )
        
        # Task heads with uncertainty
        self._build_task_heads()
        
        # Initialize weights
        self._init_weights()
    
    def _build_positional_encoding(self):
        """Build positional encoding based on config."""
        pe_type = self.config.positional_encoding
        
        if pe_type == PositionalEncodingType.SINUSOIDAL:
            self.pos_encoder = SinusoidalPositionalEncoding(
                self.config.d_model,
                max_len=self.config.input_length + 1,
                dropout=self.config.dropout,
            )
        elif pe_type == PositionalEncodingType.LEARNABLE:
            self.pos_encoder = LearnablePositionalEncoding(
                self.config.d_model,
                max_len=self.config.input_length + 1,
                dropout=self.config.dropout,
            )
        elif pe_type == PositionalEncodingType.WAVELENGTH_AWARE:
            self.pos_encoder = WavelengthAwarePositionalEncoding(
                self.config.d_model,
                max_wavelength=self.config.max_wavelength,
                dropout=self.config.dropout,
            )
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(
                self.config.d_model,
                max_len=self.config.input_length + 1,
                dropout=self.config.dropout,
            )
    
    def _build_task_heads(self):
        """Build task-specific heads with uncertainty."""
        d_model = self.config.d_model
        uncertainty = self.config.uncertainty_method
        
        # Molecule detection head
        if uncertainty == UncertaintyMethod.EVIDENTIAL:
            self.molecule_head = EvidentialHead(
                d_model, self.config.num_molecules, task="classification"
            )
        else:
            self.molecule_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(d_model // 2, self.config.num_molecules),
            )
        
        # Planet class head
        if uncertainty == UncertaintyMethod.EVIDENTIAL:
            self.planet_head = EvidentialHead(
                d_model, self.config.num_planet_classes, task="classification"
            )
        else:
            self.planet_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(d_model // 2, self.config.num_planet_classes),
            )
        
        # Habitability head with uncertainty
        if uncertainty == UncertaintyMethod.HETEROSCEDASTIC:
            self.habitability_head = HeteroscedasticHead(d_model, 1)
        elif uncertainty == UncertaintyMethod.EVIDENTIAL:
            self.habitability_head = EvidentialHead(d_model, 1, task="regression")
        else:
            self.habitability_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input spectrum (batch, length) or (batch, channels, length)
            wavelengths: Wavelength values (batch, length) or (length,)
            return_attention: If True, return attention weights
            
        Returns:
            Dictionary with predictions
        """
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, length, 1)
        elif x.dim() == 3 and x.size(1) == self.config.input_channels:
            x = x.transpose(1, 2)  # (batch, length, channels)
        
        batch_size, seq_len, _ = x.shape
        
        # Project to model dimension
        x = self.input_proj(x)  # (batch, length, d_model)
        
        # Add positional encoding
        if isinstance(self.pos_encoder, WavelengthAwarePositionalEncoding):
            x = self.pos_encoder(x, wavelengths)
        else:
            x = self.pos_encoder(x)
        
        # Add CLS token if using CLS pooling
        if self.config.pooling == "cls":
            x = self.cls_pooling.add_cls_token(x)
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Pooling
        if self.config.pooling == "cls":
            pooled = self.cls_pooling(x)
        elif self.config.pooling == "mean":
            pooled = x.mean(dim=1)
        elif self.config.pooling == "max":
            pooled = x.max(dim=1)[0]
        elif self.config.pooling == "attention":
            pooled = self.attention_pooling(x)
        else:
            pooled = x.mean(dim=1)
        
        # Task predictions
        outputs = {}
        
        # Molecule detection
        mol_out = self.molecule_head(pooled)
        if isinstance(mol_out, dict):
            outputs['molecules'] = mol_out['probabilities']
            outputs['molecule_uncertainty'] = mol_out['uncertainty']
        else:
            outputs['molecules'] = mol_out
        
        # Planet classification
        planet_out = self.planet_head(pooled)
        if isinstance(planet_out, dict):
            outputs['planet_class'] = planet_out['probabilities']
            outputs['planet_uncertainty'] = planet_out['uncertainty']
        else:
            outputs['planet_class'] = planet_out
        
        # Habitability
        hab_out = self.habitability_head(pooled)
        if isinstance(hab_out, dict):
            outputs['habitability'] = hab_out['mean']
            outputs['habitability_uncertainty'] = hab_out.get('std', hab_out.get('total_uncertainty'))
        else:
            outputs['habitability'] = hab_out.squeeze(-1)
        
        # Feature representation
        outputs['features'] = pooled
        
        return outputs
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None,
        n_samples: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using MC Dropout.
        
        Args:
            x: Input spectrum
            wavelengths: Wavelength values
            n_samples: Number of MC samples (uses config default if None)
            
        Returns:
            Predictions with uncertainty estimates
        """
        n_samples = n_samples or self.config.mc_samples
        
        if self.config.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            # Run multiple forward passes with dropout
            self.train()  # Enable dropout
            
            all_outputs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    outputs = self.forward(x, wavelengths)
                    all_outputs.append(outputs)
            
            self.eval()
            
            # Aggregate predictions
            result = {}
            
            # Molecules: average probabilities and compute variance
            mol_preds = torch.stack([
                torch.sigmoid(o['molecules']) for o in all_outputs
            ])
            result['molecules'] = mol_preds.mean(dim=0)
            result['molecule_uncertainty'] = mol_preds.std(dim=0)
            
            # Planet class
            planet_preds = torch.stack([
                F.softmax(o['planet_class'], dim=-1) for o in all_outputs
            ])
            result['planet_class'] = planet_preds.mean(dim=0)
            result['planet_uncertainty'] = planet_preds.std(dim=0).mean(dim=-1)
            
            # Habitability
            hab_preds = torch.stack([o['habitability'] for o in all_outputs])
            result['habitability'] = hab_preds.mean(dim=0)
            result['habitability_uncertainty'] = hab_preds.std(dim=0)
            
            return result
        
        else:
            # Use built-in uncertainty from evidential/heteroscedastic heads
            self.eval()
            with torch.no_grad():
                return self.forward(x, wavelengths)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Training Utilities
# =============================================================================

class EvidentialLoss(nn.Module):
    """
    Loss function for evidential deep learning.
    """
    
    def __init__(self, task: str = "classification", lambda_reg: float = 0.1):
        super().__init__()
        self.task = task
        self.lambda_reg = lambda_reg
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.task == "classification":
            # Dirichlet loss
            alpha = outputs['alpha']
            S = alpha.sum(dim=-1, keepdim=True)
            
            # Expected cross-entropy
            loss = torch.sum(
                targets * (torch.digamma(S) - torch.digamma(alpha)),
                dim=-1
            )
            
            # KL divergence regularization
            alpha_tilde = targets + (1 - targets) * alpha
            kl = self._kl_divergence(alpha_tilde)
            
            return (loss + self.lambda_reg * kl).mean()
        
        else:
            # NIG loss for regression
            gamma, nu, alpha, beta = outputs['nig_params']
            
            # Negative log-likelihood
            nll = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(beta)
            nll += (alpha + 0.5) * torch.log(nu * (targets - gamma)**2 + beta)
            nll += torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
            
            # Regularization
            reg = (2 * nu + alpha) * torch.abs(targets - gamma)
            
            return (nll + self.lambda_reg * reg).mean()
    
    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL divergence from uniform Dirichlet."""
        K = alpha.size(-1)
        alpha0 = alpha.sum(dim=-1)
        
        kl = torch.lgamma(alpha0) - torch.lgamma(torch.tensor(K, dtype=alpha.dtype))
        kl -= torch.lgamma(alpha).sum(dim=-1)
        kl += ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(alpha0.unsqueeze(-1)))).sum(dim=-1)
        
        return kl


class TransformerTrainer:
    """
    Training pipeline for the Transformer model.
    """
    
    def __init__(
        self,
        model: SpectralTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_epochs: int = 100,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * max_epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + math.cos(
                    math.pi * (step - warmup_steps) / (total_steps - warmup_steps)
                ))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        
        # Loss functions
        self.molecule_loss = nn.BCEWithLogitsLoss()
        self.planet_loss = nn.CrossEntropyLoss()
        self.habitability_loss = nn.MSELoss()
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Tracking
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
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
        
        with torch.amp.autocast('cuda'):
            outputs = self.model(spectrum, wavelengths)
            
            loss = (
                self.molecule_loss(outputs['molecules'], targets['molecules']) +
                self.planet_loss(outputs['planet_class'], targets['planet_class']) +
                self.habitability_loss(outputs['habitability'], targets['habitability'])
            )
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
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
            
            loss = (
                self.molecule_loss(outputs['molecules'], targets['molecules']) +
                self.planet_loss(outputs['planet_class'], targets['planet_class']) +
                self.habitability_loss(outputs['habitability'], targets['habitability'])
            )
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            
            for batch in self.train_loader:
                loss = self.train_step(batch)
                epoch_loss += loss
            
            train_loss = epoch_loss / len(self.train_loader)
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_transformer.pt')
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs} - "
                    f"Train: {train_loss:.4f}, Val: {val_loss:.4f}"
                )
        
        return self.history


# =============================================================================
# Convenience Functions
# =============================================================================

def create_spectral_transformer(
    input_length: int = 512,
    pretrained_path: Optional[str] = None,
    **config_kwargs,
) -> SpectralTransformer:
    """
    Create Transformer model with optional pretrained weights.
    """
    config = TransformerConfig(input_length=input_length, **config_kwargs)
    model = SpectralTransformer(config)
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
    
    return model


@torch.no_grad()
def predict_with_transformer(
    model: SpectralTransformer,
    spectrum: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    with_uncertainty: bool = True,
) -> Dict[str, Any]:
    """
    Make predictions with uncertainty for a single spectrum.
    """
    device = next(model.parameters()).device
    
    x = torch.from_numpy(spectrum).float().unsqueeze(0).to(device)
    
    wl = None
    if wavelengths is not None:
        wl = torch.from_numpy(wavelengths).float().to(device)
    
    if with_uncertainty:
        outputs = model.predict_with_uncertainty(x, wl)
    else:
        model.eval()
        outputs = model(x, wl)
        outputs['molecules'] = torch.sigmoid(outputs['molecules'])
        outputs['planet_class'] = F.softmax(outputs['planet_class'], dim=-1)
    
    # Process outputs
    mol_probs = outputs['molecules'][0].cpu().numpy()
    planet_probs = outputs['planet_class'][0].cpu().numpy()
    habitability = outputs['habitability'][0].cpu().item()
    
    result = {
        'detected_molecules': [
            MOLECULE_NAMES[i] for i, p in enumerate(mol_probs) if p > 0.5
        ],
        'molecule_probabilities': {
            MOLECULE_NAMES[i]: float(p) for i, p in enumerate(mol_probs)
        },
        'planet_class': PLANET_CLASS_NAMES[planet_probs.argmax()],
        'planet_probabilities': {
            PLANET_CLASS_NAMES[i]: float(p) for i, p in enumerate(planet_probs)
        },
        'habitability_score': habitability,
    }
    
    if with_uncertainty and 'molecule_uncertainty' in outputs:
        mol_unc = outputs['molecule_uncertainty'][0].cpu().numpy()
        result['molecule_uncertainty'] = {
            MOLECULE_NAMES[i]: float(u) for i, u in enumerate(mol_unc)
        }
        result['habitability_uncertainty'] = outputs['habitability_uncertainty'][0].cpu().item()
    
    return result


# =============================================================================
# Main (Demo)
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("Spectral Transformer Demo")
    print("=" * 70)
    
    # Create model
    config = TransformerConfig(
        input_length=512,
        d_model=256,
        n_heads=8,
        n_layers=6,
        positional_encoding=PositionalEncodingType.WAVELENGTH_AWARE,
        uncertainty_method=UncertaintyMethod.MC_DROPOUT,
    )
    
    model = SpectralTransformer(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 512
    
    x = torch.randn(batch_size, seq_len)
    wavelengths = torch.linspace(0.6, 5.0, seq_len)
    
    print("\n--- Forward Pass ---")
    outputs = model(x, wavelengths)
    
    print(f"Molecules shape: {outputs['molecules'].shape}")
    print(f"Planet class shape: {outputs['planet_class'].shape}")
    print(f"Habitability shape: {outputs['habitability'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    
    # Test uncertainty estimation
    print("\n--- Uncertainty Estimation (MC Dropout) ---")
    uncertainty_outputs = model.predict_with_uncertainty(x[:1], wavelengths, n_samples=10)
    
    print(f"Molecule probabilities: {uncertainty_outputs['molecules'].shape}")
    print(f"Molecule uncertainty: {uncertainty_outputs['molecule_uncertainty'].shape}")
    print(f"Habitability: {uncertainty_outputs['habitability'].item():.4f}")
    print(f"Habitability uncertainty: {uncertainty_outputs['habitability_uncertainty'].item():.4f}")
    
    # Test wavelength-aware encoding
    print("\n--- Wavelength-Aware Encoding ---")
    encoder = WavelengthAwarePositionalEncoding(d_model=256)
    
    test_input = torch.randn(2, 512, 256)
    test_wl = torch.linspace(0.6, 5.0, 512)
    
    encoded = encoder(test_input, test_wl)
    print(f"Input shape: {test_input.shape}")
    print(f"Encoded shape: {encoded.shape}")
    
    # Single prediction
    print("\n--- Single Prediction ---")
    sample_spectrum = np.random.randn(512).astype(np.float32)
    sample_wl = np.linspace(0.6, 5.0, 512).astype(np.float32)
    
    result = predict_with_transformer(
        model, sample_spectrum, sample_wl, with_uncertainty=True
    )
    
    print(f"Detected molecules: {result['detected_molecules']}")
    print(f"Planet class: {result['planet_class']}")
    print(f"Habitability: {result['habitability_score']:.3f}")
    if 'habitability_uncertainty' in result:
        print(f"Habitability uncertainty: {result['habitability_uncertainty']:.3f}")
