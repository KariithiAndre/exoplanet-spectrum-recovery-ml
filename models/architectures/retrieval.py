"""
Atmospheric Retrieval Network

Neural network for end-to-end atmospheric parameter estimation from spectra.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class AttentionPooling(nn.Module):
    """Attention-based pooling layer for sequence aggregation."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Pooled tensor of shape (batch, features)
        """
        attn_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(attn_weights * x, dim=1)


class RetrievalNetwork(nn.Module):
    """
    Neural network for atmospheric parameter retrieval.
    
    This model takes a transmission spectrum as input and predicts
    atmospheric parameters such as:
    - Temperature profile
    - Molecular abundances (H2O, CO2, CH4, CO, etc.)
    - Cloud properties
    - Reference pressure/radius
    
    Architecture:
    - Feature extractor: 1D CNN for spectral feature extraction
    - Sequence modeling: Bidirectional LSTM for capturing wavelength dependencies
    - Attention pooling: Weighted aggregation of sequence features
    - Parameter heads: Separate prediction heads for each parameter group
    """
    
    # Default atmospheric parameters to predict
    DEFAULT_PARAMETERS = {
        "temperature": 1,           # Isothermal temperature (K)
        "log_h2o": 1,              # log10 H2O mixing ratio
        "log_co2": 1,              # log10 CO2 mixing ratio
        "log_ch4": 1,              # log10 CH4 mixing ratio
        "log_co": 1,               # log10 CO mixing ratio
        "log_cloud_top": 1,        # log10 cloud top pressure
        "cloud_fraction": 1,       # Cloud coverage fraction
        "reference_radius": 1,     # Planet radius at reference pressure
    }
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        output_params: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_params = output_params or self.DEFAULT_PARAMETERS
        self.total_outputs = sum(self.output_params.values())
        
        # Feature extractor (1D CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Sequence modeling (Bidirectional LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_dim * 2)  # *2 for bidirectional
        
        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Parameter prediction heads
        self.param_heads = nn.ModuleDict()
        for param_name, param_dim in self.output_params.items():
            self.param_heads[param_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, param_dim),
            )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, self.total_outputs),
            nn.Softplus(),  # Ensure positive uncertainties
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the retrieval network.
        
        Args:
            x: Input spectrum of shape (batch, channels, wavelength_points)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary mapping parameter names to predicted values
        """
        # Extract features
        features = self.feature_extractor(x)  # (B, hidden_dim, L/4)
        
        # Transpose for LSTM: (B, L, C)
        features = features.permute(0, 2, 1)
        
        # Sequence modeling
        lstm_out, _ = self.lstm(features)  # (B, L, hidden_dim*2)
        
        # Attention pooling
        pooled = self.attention_pool(lstm_out)  # (B, hidden_dim*2)
        
        # Shared features
        shared = self.shared_fc(pooled)  # (B, hidden_dim)
        
        # Predict each parameter
        predictions = {}
        for param_name, head in self.param_heads.items():
            predictions[param_name] = head(shared)
        
        # Add uncertainty if requested
        if return_uncertainty:
            predictions["uncertainty"] = self.uncertainty_head(shared)
        
        return predictions
    
    def predict_with_samples(
        self,
        x: torch.Tensor,
        num_samples: int = 100,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Predict with Monte Carlo dropout for uncertainty estimation.
        
        Args:
            x: Input spectrum
            num_samples: Number of MC samples
            
        Returns:
            Tuple of (mean predictions, std predictions)
        """
        self.train()  # Enable dropout
        
        samples = {name: [] for name in self.output_params.keys()}
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, return_uncertainty=False)
                for name in self.output_params.keys():
                    samples[name].append(pred[name])
        
        means = {}
        stds = {}
        for name in self.output_params.keys():
            stacked = torch.stack(samples[name], dim=0)
            means[name] = stacked.mean(dim=0)
            stds[name] = stacked.std(dim=0)
        
        self.eval()
        return means, stds


class FeatureDetector(nn.Module):
    """
    CNN for detecting molecular absorption features in spectra.
    
    This model identifies and localizes absorption features from
    specific molecules in the spectrum, providing both classification
    and localization outputs.
    """
    
    MOLECULES = ["H2O", "CO2", "CH4", "CO", "NH3", "Na", "K", "TiO", "VO"]
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.num_classes = num_classes or len(self.MOLECULES)
        
        # Feature backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Classification head (global pooling -> FC)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.num_classes),
            nn.Sigmoid(),  # Multi-label classification
        )
        
        # Localization head (per-position scores)
        self.localizer = nn.Conv1d(hidden_dim, self.num_classes, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for feature detection.
        
        Args:
            x: Input spectrum of shape (batch, channels, wavelength_points)
            return_maps: Whether to return localization maps
            
        Returns:
            Dictionary with 'detection' (presence scores) and optionally 'localization'
        """
        features = self.backbone(x)
        
        # Global detection
        detection = self.classifier(features)
        
        output = {"detection": detection}
        
        if return_maps:
            # Per-position localization
            localization = torch.sigmoid(self.localizer(features))
            output["localization"] = localization
        
        return output
