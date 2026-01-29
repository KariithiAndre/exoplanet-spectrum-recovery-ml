"""
Spectrum Denoiser Neural Network

A deep learning model for denoising exoplanet transmission spectra.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SpectrumDenoiser(nn.Module):
    """
    Convolutional neural network for spectral noise reduction.
    
    Architecture:
    - Encoder: Downsampling convolutional layers
    - Bottleneck: Residual blocks for feature processing
    - Decoder: Upsampling layers to reconstruct spectrum
    
    Args:
        input_channels: Number of input channels (default: 1)
        base_channels: Base number of feature channels (default: 64)
        num_residual_blocks: Number of residual blocks in bottleneck (default: 4)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 64,
        num_residual_blocks: int = 4,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck (residual blocks)
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(base_channels * 4) for _ in range(num_residual_blocks)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, input_channels, kernel_size=7, padding=3),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the denoiser.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            
        Returns:
            Denoised spectrum of same shape as input
        """
        # Store input for skip connection
        identity = x
        
        # Encode
        encoded = self.encoder(x)
        
        # Process through bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # Decode
        decoded = self.decoder(bottleneck)
        
        # Adjust size if needed (due to stride operations)
        if decoded.shape[-1] != identity.shape[-1]:
            decoded = F.interpolate(decoded, size=identity.shape[-1], mode='linear', align_corners=False)
        
        # Residual learning: predict the noise to subtract
        return identity - decoded


class SpectrumDenoiserV2(nn.Module):
    """
    Transformer-based spectrum denoiser with attention mechanisms.
    
    This model uses self-attention to capture long-range dependencies
    in spectral features, which is particularly useful for identifying
    broad absorption bands and correlations across wavelengths.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_dim),
        )
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer denoiser.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
               or (batch, input_dim, sequence_length) if channel-first
            
        Returns:
            Denoised spectrum of same shape as input
        """
        # Handle channel-first input
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
            channel_first = True
        else:
            channel_first = False
        
        identity = x
        
        # Project to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transform
        x = self.transformer(x)
        
        # Project back to input dimension
        noise_estimate = self.output_proj(x)
        
        # Residual learning
        output = identity - noise_estimate
        
        # Restore channel-first format if needed
        if channel_first:
            output = output.permute(0, 2, 1)
        
        return output


def create_denoiser(model_type: str = "v1", **kwargs) -> nn.Module:
    """
    Factory function to create a denoiser model.
    
    Args:
        model_type: Either "v1" (CNN) or "v2" (Transformer)
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Instantiated denoiser model
    """
    if model_type == "v1":
        return SpectrumDenoiser(**kwargs)
    elif model_type == "v2":
        return SpectrumDenoiserV2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
