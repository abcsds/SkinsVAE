#!/usr/bin/env python3
"""
Utility script for exploring trained VAE models and generating new Minecraft skins.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
from minecraft_skin_vae import ConvVAE


class VAEExplorer:
    """Utility class for exploring trained VAE models."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize the VAE explorer.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        self.latent_dim = checkpoint['latent_dim']
        self.model = ConvVAE(latent_dim=self.latent_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully! Latent dimension: {self.latent_dim}")
    
    def generate_random_skins(self, num_skins: int = 8, save_dir: str = "generated_skins"):
        """
        Generate random skins by sampling from the latent space.
        
        Args:
            num_skins: Number of skins to generate
            save_dir: Directory to save generated skins
        """
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_skins, self.latent_dim).to(self.device)
            
            # Generate skins
            generated = self.model.decode(z)
            
            # Convert to numpy and save
            generated = generated.cpu().numpy()
            
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()
            
            for i in range(num_skins):
                # Convert from RGBA to RGB for display
                skin_rgb = generated[i][:3]  # Take only RGB channels
                skin_rgb = np.transpose(skin_rgb, (1, 2, 0))
                skin_rgb = np.clip(skin_rgb, 0, 1)
                
                # Display
                axes[i].imshow(skin_rgb)
                axes[i].set_title(f'Generated Skin {i+1}')
                axes[i].axis('off')
                
                # Save as PNG
                skin_rgba = generated[i]  # Full RGBA
                skin_rgba = np.transpose(skin_rgba, (1, 2, 0))
                skin_rgba = np.clip(skin_rgba * 255, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(skin_rgba, 'RGBA')
                img.save(os.path.join(save_dir, f'generated_skin_{i+1}.png'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'generated_skins_overview.png'), dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Generated {num_skins} skins saved to {save_dir}/")
    
    def interpolate_between_latents(self, z1: np.ndarray, z2: np.ndarray, steps: int = 10, 
                                  save_dir: str = "interpolation"):
        """
        Interpolate between two latent vectors.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            steps: Number of interpolation steps
            save_dir: Directory to save interpolation results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            # Create interpolation
            alphas = np.linspace(0, 1, steps)
            interpolated_z = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                interpolated_z.append(z_interp)
            
            interpolated_z = np.stack(interpolated_z)
            interpolated_z = torch.tensor(interpolated_z, dtype=torch.float32).to(self.device)
            
            # Generate interpolated skins
            generated = self.model.decode(interpolated_z)
            generated = generated.cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, steps//2, figsize=(15, 6))
            axes = axes.flatten()
            
            for i in range(steps):
                skin_rgb = generated[i][:3]
                skin_rgb = np.transpose(skin_rgb, (1, 2, 0))
                skin_rgb = np.clip(skin_rgb, 0, 1)
                
                axes[i].imshow(skin_rgb)
                axes[i].set_title(f'Step {i+1} (α={alphas[i]:.2f})')
                axes[i].axis('off')
                
                # Save individual frame
                skin_rgba = generated[i]
                skin_rgba = np.transpose(skin_rgba, (1, 2, 0))
                skin_rgba = np.clip(skin_rgba * 255, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(skin_rgba, 'RGBA')
                img.save(os.path.join(save_dir, f'interpolation_step_{i+1:02d}.png'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'interpolation_overview.png'), dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Interpolation saved to {save_dir}/")
    
    def encode_skin(self, skin_path: str):
        """
        Encode a skin image to latent space.
        
        Args:
            skin_path: Path to skin PNG file
            
        Returns:
            Latent vector (numpy array)
        """
        # Load and preprocess image
        img = Image.open(skin_path).convert('RGBA')
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.NEAREST)
        
        # Convert to tensor
        img_array = np.array(img) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            mu, logvar = self.model.encode(img_tensor)
            return mu.cpu().numpy().flatten()
    
    def reconstruct_skin(self, skin_path: str, save_path: str = None):
        """
        Reconstruct a skin through the VAE.
        
        Args:
            skin_path: Path to input skin PNG file
            save_path: Path to save reconstructed skin (optional)
        """
        # Load and preprocess image
        img = Image.open(skin_path).convert('RGBA')
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.NEAREST)
        
        # Convert to tensor
        img_array = np.array(img) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            # Reconstruct
            recon, mu, logvar = self.model(img_tensor)
            
            # Convert back to image
            recon = recon.cpu().numpy().squeeze()
            recon = np.transpose(recon, (1, 2, 0))
            recon = np.clip(recon * 255, 0, 255).astype(np.uint8)
            
            # Display comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            ax1.imshow(img_array[:, :, :3])  # Original (RGB only)
            ax1.set_title('Original')
            ax1.axis('off')
            
            ax2.imshow(recon[:, :, :3])  # Reconstruction (RGB only)
            ax2.set_title('Reconstructed')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Save if requested
            if save_path:
                recon_img = Image.fromarray(recon, 'RGBA')
                recon_img.save(save_path)
                print(f"Reconstructed skin saved to {save_path}")
            
            return mu.cpu().numpy().flatten()


def main():
    """Main function for VAE exploration utilities."""
    parser = argparse.ArgumentParser(description='Explore trained Minecraft Skin VAE')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['generate', 'interpolate', 'reconstruct', 'encode'],
                       default='generate', help='Exploration mode')
    parser.add_argument('--num_skins', type=int, default=8, help='Number of skins to generate')
    parser.add_argument('--skin_path', type=str, help='Path to skin file (for reconstruct/encode modes)')
    parser.add_argument('--output_dir', type=str, default='exploration_output', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = VAEExplorer(args.checkpoint, args.device)
    
    if args.mode == 'generate':
        print(f"Generating {args.num_skins} random skins...")
        explorer.generate_random_skins(args.num_skins, args.output_dir)
    
    elif args.mode == 'interpolate':
        print("Generating interpolation between two random latent vectors...")
        # Generate two random latent vectors
        z1 = np.random.randn(explorer.latent_dim)
        z2 = np.random.randn(explorer.latent_dim)
        explorer.interpolate_between_latents(z1, z2, steps=10, save_dir=args.output_dir)
    
    elif args.mode == 'reconstruct':
        if not args.skin_path:
            print("Error: --skin_path required for reconstruct mode")
            return
        
        print(f"Reconstructing skin from {args.skin_path}...")
        save_path = os.path.join(args.output_dir, 'reconstructed_skin.png')
        os.makedirs(args.output_dir, exist_ok=True)
        latent = explorer.reconstruct_skin(args.skin_path, save_path)
        print(f"Latent vector: {latent}")
    
    elif args.mode == 'encode':
        if not args.skin_path:
            print("Error: --skin_path required for encode mode")
            return
        
        print(f"Encoding skin from {args.skin_path}...")
        latent = explorer.encode_skin(args.skin_path)
        print(f"Latent vector: {latent}")
        
        # Save latent vector
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, 'latent_vector.npy'), latent)
        print(f"Latent vector saved to {args.output_dir}/latent_vector.npy")


if __name__ == "__main__":
    main()
