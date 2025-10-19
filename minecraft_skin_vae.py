#!/usr/bin/env python3
"""
Minecraft Skins Variational Autoencoder (VAE)

This script implements a convolutional VAE to learn a latent representation
of Minecraft skins (64x64 RGBA images).

Features:
- Convolutional encoder/decoder architecture
- Configurable latent dimension (default: 3 for visualization)
- TensorBoard logging for training metrics and visualizations
- GPU acceleration (CUDA)
- Model checkpointing and resuming
- Batch training with data augmentation
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from typing import Tuple, Dict, Any


class MinecraftSkinDataset(Dataset):
    """Dataset class for loading Minecraft skin images."""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Directory containing PNG skin files
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG files found in {data_dir}")
        
        print(f"Found {len(self.image_paths)} skin images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image and ensure it's RGBA
        image = Image.open(img_path).convert('RGBA')
        
        # Verify size
        if image.size != (64, 64):
            image = image.resize((64, 64), Image.NEAREST)
        
        if self.transform:
            image = self.transform(image)
        
        return image


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for Minecraft skins."""
    
    def __init__(self, latent_dim: int = 3, input_channels: int = 4):
        """
        Args:
            latent_dim: Dimension of the latent space
            input_channels: Number of input channels (4 for RGBA)
        """
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            # 64x64x4 -> 32x32x32
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 16x16x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate flattened size after convolution
        self.flatten_size = 4 * 4 * 256
        
        # Latent space layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 16x16x64 -> 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 64x64x4
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)  # Reshape for deconvolution
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
             logvar: torch.Tensor, beta: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    VAE loss function (reconstruction + KL divergence).
    
    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    
    Returns:
        Dictionary containing total loss and components
    """
    # Reconstruction loss (binary cross entropy)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


class VAETrainer:
    """Training class for the VAE."""
    
    def __init__(self, model: ConvVAE, device: torch.device, log_dir: str = "runs"):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"skin_vae_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"TensorBoard logs will be saved to: {self.log_dir}")
        print(f"View with: tensorboard --logdir={self.log_dir}")
    
    def train(self, train_loader: DataLoader, num_epochs: int = 100, 
              learning_rate: float = 1e-3, beta: float = 1.0,
              checkpoint_dir: str = "checkpoints", save_freq: int = 10):
        """
        Train the VAE.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            beta: Beta parameter for beta-VAE
            checkpoint_dir: Directory to save checkpoints
            save_freq: Frequency to save checkpoints (epochs)
        """
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar = self.model(data)
                
                # Calculate loss
                loss_dict = vae_loss(recon_batch, data, mu, logvar, beta)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
                
                # Log to TensorBoard
                if batch_idx % 10 == 0:
                    self.writer.add_scalar('Loss/Batch_Total', loss.item(), global_step)
                    self.writer.add_scalar('Loss/Batch_Reconstruction', 
                                         loss_dict['recon_loss'].item(), global_step)
                    self.writer.add_scalar('Loss/Batch_KL', 
                                         loss_dict['kl_loss'].item(), global_step)
                
                global_step += 1
            
            # Average losses for epoch
            avg_loss = total_loss / len(train_loader.dataset)
            avg_recon_loss = total_recon_loss / len(train_loader.dataset)
            avg_kl_loss = total_kl_loss / len(train_loader.dataset)
            
            # Log epoch metrics
            self.writer.add_scalar('Loss/Epoch_Total', avg_loss, epoch)
            self.writer.add_scalar('Loss/Epoch_Reconstruction', avg_recon_loss, epoch)
            self.writer.add_scalar('Loss/Epoch_KL', avg_kl_loss, epoch)
            
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Loss: {avg_loss:.4f}, '
                  f'Recon: {avg_recon_loss:.4f}, '
                  f'KL: {avg_kl_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'vae_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path, epoch, optimizer, avg_loss)
            
            # Log reconstructions and latent space visualizations
            if epoch % 5 == 0:
                self.log_reconstructions(data[:8], epoch)
                self.log_latent_space(train_loader, epoch)
    
    def log_reconstructions(self, sample_batch: torch.Tensor, epoch: int):
        """Log original and reconstructed images to TensorBoard."""
        self.model.eval()
        with torch.no_grad():
            recon_batch, _, _ = self.model(sample_batch)
            
            # Convert to numpy for visualization (take only RGB channels)
            originals = sample_batch[:, :3].cpu().numpy()
            reconstructions = recon_batch[:, :3].cpu().numpy()
            
            # Create comparison grid
            fig, axes = plt.subplots(2, len(originals), figsize=(15, 4))
            
            for i in range(len(originals)):
                # Original
                axes[0, i].imshow(np.transpose(originals[i], (1, 2, 0)))
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstruction
                axes[1, i].imshow(np.transpose(reconstructions[i], (1, 2, 0)))
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            self.writer.add_figure('Reconstructions', fig, epoch)
            plt.close(fig)
        
        self.model.train()
    
    def log_latent_space(self, data_loader: DataLoader, epoch: int, max_samples: int = 1000):
        """Log latent space visualization to TensorBoard with clustering and dimensionality reduction."""
        self.model.eval()
        latent_vectors = []
        
        with torch.no_grad():
            sample_count = 0
            for data in data_loader:
                if sample_count >= max_samples:
                    break
                
                data = data.to(self.device)
                mu, _ = self.model.encode(data)
                latent_vectors.append(mu.cpu().numpy())
                sample_count += len(data)
            
            latent_vectors = np.concatenate(latent_vectors, axis=0)[:max_samples]
            
            # Apply clustering and dimensionality reduction for high-dimensional latent spaces
            self._visualize_latent_space_with_clustering(latent_vectors, epoch)
        
        self.model.train()
    
    def _visualize_latent_space_with_clustering(self, latent_vectors: np.ndarray, epoch: int):
        """Apply clustering and dimensionality reduction for latent space visualization."""
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        import umap
        import seaborn as sns
        
        # Apply K-means clustering
        n_clusters = min(8, len(latent_vectors) // 10)  # Adaptive number of clusters
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(latent_vectors)
        else:
            cluster_labels = np.zeros(len(latent_vectors))
        
        # Apply dimensionality reduction techniques
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. t-SNE visualization
        if len(latent_vectors) > 5:  # Need minimum samples for t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)//4))
            tsne_result = tsne.fit_transform(latent_vectors)
            
            scatter = axes[0, 0].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                       c=cluster_labels, cmap='tab10', alpha=0.7, s=20)
            axes[0, 0].set_title(f't-SNE Latent Space (Epoch {epoch})')
            axes[0, 0].set_xlabel('t-SNE 1')
            axes[0, 0].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        # 2. UMAP visualization
        if len(latent_vectors) > 10:  # UMAP needs more samples
            try:
                umap_reducer = umap.UMAP(n_components=2, random_state=42, 
                                       n_neighbors=min(15, len(latent_vectors)//2))
                umap_result = umap_reducer.fit_transform(latent_vectors)
                
                scatter = axes[0, 1].scatter(umap_result[:, 0], umap_result[:, 1], 
                                           c=cluster_labels, cmap='tab10', alpha=0.7, s=20)
                axes[0, 1].set_title(f'UMAP Latent Space (Epoch {epoch})')
                axes[0, 1].set_xlabel('UMAP 1')
                axes[0, 1].set_ylabel('UMAP 2')
                plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'UMAP failed: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('UMAP (Failed)')
        
        # 3. Latent space statistics
        axes[1, 0].hist(latent_vectors.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title(f'Latent Space Distribution (Epoch {epoch})')
        axes[1, 0].set_xlabel('Latent Value')
        axes[1, 0].set_ylabel('Density')
        
        # 4. Cluster centers heatmap
        if n_clusters > 1:
            cluster_centers = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    center = latent_vectors[cluster_mask].mean(axis=0)
                    cluster_centers.append(center)
            
            if cluster_centers:
                cluster_centers = np.array(cluster_centers)
                im = axes[1, 1].imshow(cluster_centers, aspect='auto', cmap='viridis')
                axes[1, 1].set_title(f'Cluster Centers (Epoch {epoch})')
                axes[1, 1].set_xlabel('Latent Dimension')
                axes[1, 1].set_ylabel('Cluster')
                plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data for clustering', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        self.writer.add_figure('Latent_Space_Analysis', fig, epoch)
        plt.close(fig)
        
        # Log cluster information
        if n_clusters > 1:
            unique, counts = np.unique(cluster_labels, return_counts=True)
            cluster_info = dict(zip(unique, counts))
            for cluster_id, count in cluster_info.items():
                self.writer.add_scalar(f'Clustering/Cluster_{cluster_id}_Size', count, epoch)
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer: optim.Optimizer, loss: float):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'latent_dim': self.model.latent_dim,
        }, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str, optimizer: optim.Optimizer = None):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resumed from epoch {epoch+1}, loss: {loss:.4f}")
        
        return epoch, loss


def create_data_transforms():
    """Create data transforms for training."""
    return transforms.Compose([
        transforms.ToTensor(),
        # Optional: Add data augmentation
        # transforms.RandomHorizontalFlip(p=0.5),
    ])


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Minecraft Skin VAE')
    parser.add_argument('--data_dir', type=str, default='/home/beto/MCSkins/skins/',
                       help='Directory containing skin PNG files')
    parser.add_argument('--latent_dim', type=int, default=3,
                       help='Latent space dimensionality')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for beta-VAE')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Frequency to save checkpoints (epochs)')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create dataset and data loader
    transform = create_data_transforms()
    dataset = MinecraftSkinDataset(args.data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Create model
    model = ConvVAE(latent_dim=args.latent_dim)
    print(f"Model created with latent dimension: {args.latent_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = VAETrainer(model, device)
    
    # Setup optimizer for potential checkpoint loading
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, _ = trainer.load_checkpoint(args.resume, optimizer)
            start_epoch += 1  # Start from next epoch
        else:
            print(f"Checkpoint file not found: {args.resume}")
    
    # Train the model
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        num_epochs=args.num_epochs - start_epoch,
        learning_rate=args.learning_rate,
        beta=args.beta,
        checkpoint_dir=args.checkpoint_dir,
        save_freq=args.save_freq
    )
    
    print("Training completed!")
    print(f"TensorBoard logs saved in: {trainer.log_dir}")
    print(f"To view: tensorboard --logdir={trainer.log_dir}")


if __name__ == "__main__":
    main()
