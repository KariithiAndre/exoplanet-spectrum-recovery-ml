"""
Training script for spectrum denoiser models.

This script handles:
- Data loading from synthetic spectra
- Model training with configurable hyperparameters
- Validation and early stopping
- Model checkpointing
- Training visualization
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.architectures.denoiser import create_denoiser


class SyntheticSpectrumDataset(Dataset):
    """
    Dataset for loading synthetic spectra with noise.
    
    Expects data in format:
    - clean_spectra.npy: Clean reference spectra (N, wavelength_points)
    - noisy_spectra.npy: Noisy versions of spectra (N, wavelength_points)
    - wavelength.npy: Wavelength grid (wavelength_points,)
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        noise_level: Optional[float] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load data
        self.clean = np.load(self.data_dir / "clean_spectra.npy")
        
        if (self.data_dir / "noisy_spectra.npy").exists():
            self.noisy = np.load(self.data_dir / "noisy_spectra.npy")
        else:
            # Generate noisy version on the fly
            noise_level = noise_level or 0.05
            noise = np.random.normal(0, noise_level * self.clean.std(), self.clean.shape)
            self.noisy = self.clean + noise
        
        self.wavelength = np.load(self.data_dir / "wavelength.npy")
        
    def __len__(self) -> int:
        return len(self.clean)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy = torch.FloatTensor(self.noisy[idx]).unsqueeze(0)  # Add channel dim
        clean = torch.FloatTensor(self.clean[idx]).unsqueeze(0)
        
        if self.transform:
            noisy, clean = self.transform(noisy, clean)
        
        return noisy, clean


class RandomNoiseAugmentation:
    """Data augmentation by adding random noise levels."""
    
    def __init__(self, min_snr: float = 10, max_snr: float = 100):
        self.min_snr = min_snr
        self.max_snr = max_snr
    
    def __call__(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random SNR for this sample
        snr = np.random.uniform(self.min_snr, self.max_snr)
        noise_std = clean.std() / snr
        noise = torch.randn_like(clean) * noise_std
        
        return clean + noise, clean


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * noisy.size(0)
    
    return total_loss / len(dataloader.dataset)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_snr_improvement = 0.0
    
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item() * noisy.size(0)
            
            # Calculate SNR improvement
            input_error = (noisy - clean).pow(2).mean(dim=-1)
            output_error = (output - clean).pow(2).mean(dim=-1)
            snr_improvement = 10 * torch.log10(input_error / (output_error + 1e-10))
            total_snr_improvement += snr_improvement.sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_snr_improvement = total_snr_improvement / len(dataloader.dataset)
    
    return avg_loss, avg_snr_improvement


def main(args):
    """Main training function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create model
    model = create_denoiser(model_type=args.model_type)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    transform = RandomNoiseAugmentation(min_snr=args.min_snr, max_snr=args.max_snr)
    dataset = SyntheticSpectrumDataset(args.data_dir, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Tensorboard
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, snr_improvement = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/snr_improvement_db", snr_improvement, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"SNR Improvement: {snr_improvement:.2f} dB")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "snr_improvement": snr_improvement,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Regular checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    writer.close()
    print(f"Training complete. Best model saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spectrum denoiser model")
    
    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="./models/checkpoints",
                        help="Output directory for checkpoints")
    
    # Model
    parser.add_argument("--model-type", type=str, default="v1",
                        choices=["v1", "v2"], help="Model architecture version")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Checkpoint frequency (epochs)")
    
    # Data augmentation
    parser.add_argument("--min-snr", type=float, default=10)
    parser.add_argument("--max-snr", type=float, default=100)
    
    # Hardware
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
