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
    
    def analyze_latent_space_from_dataset(self, dataset_path: str, max_samples: int = 1000, 
                                         save_dir: str = "latent_analysis"):
        """
        Analyze the latent space using clustering and dimensionality reduction on a dataset.
        
        Args:
            dataset_path: Path to directory containing skin images
            max_samples: Maximum number of samples to analyze
            save_dir: Directory to save analysis results
        """
        from minecraft_skin_vae import MinecraftSkinDataset, create_data_transforms
        from torch.utils.data import DataLoader
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import umap
        import seaborn as sns
        
        print(f"Analyzing latent space from dataset: {dataset_path}")
        
        # Create dataset and dataloader
        transform = create_data_transforms()
        dataset = MinecraftSkinDataset(dataset_path, transform=transform)
        
        # Limit samples if needed
        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Extract latent representations
        print("Extracting latent representations...")
        latent_vectors = []
        original_images = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                data = data.to(self.device)
                mu, _ = self.model.encode(data)
                latent_vectors.append(mu.cpu().numpy())
                
                # Store some original images for visualization
                if batch_idx < 3:  # Store first few batches
                    original_images.append(data.cpu().numpy())
        
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        original_images = np.concatenate(original_images, axis=0) if original_images else None
        
        print(f"Analyzing {len(latent_vectors)} latent vectors of dimension {self.latent_dim}")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Apply clustering algorithms
        clusters_info = self._apply_clustering(latent_vectors, save_dir)
        
        # Apply dimensionality reduction and visualization
        self._visualize_latent_space(latent_vectors, clusters_info, save_dir, original_images)
        
        # Generate cluster representative skins
        self._generate_cluster_representatives(latent_vectors, clusters_info, save_dir)
        
        print(f"Analysis complete! Results saved to {save_dir}/")
    
    def _apply_clustering(self, latent_vectors: np.ndarray, save_dir: str):
        """Apply various clustering algorithms to latent vectors."""
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        
        print("Applying clustering algorithms...")
        clusters_info = {}
        
        # K-means clustering with different k values
        silhouette_scores = []
        k_values = range(2, min(11, len(latent_vectors) // 5))
        
        best_k = 2
        best_score = -1
        
        for k in k_values:
            if k >= len(latent_vectors):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(latent_vectors)
            score = silhouette_score(latent_vectors, labels)
            silhouette_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Use best k for final clustering
        print(f"Best K-means: k={best_k} (silhouette score: {best_score:.3f})")
        kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_best.fit_predict(latent_vectors)
        
        clusters_info['kmeans'] = {
            'labels': kmeans_labels,
            'centers': kmeans_best.cluster_centers_,
            'n_clusters': best_k,
            'silhouette_score': best_score
        }
        
        # DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(latent_vectors)
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            
            if n_clusters_dbscan > 1:
                dbscan_score = silhouette_score(latent_vectors, dbscan_labels)
                clusters_info['dbscan'] = {
                    'labels': dbscan_labels,
                    'n_clusters': n_clusters_dbscan,
                    'silhouette_score': dbscan_score
                }
                print(f"DBSCAN: {n_clusters_dbscan} clusters (silhouette score: {dbscan_score:.3f})")
            else:
                print("DBSCAN: Failed to find meaningful clusters")
        except Exception as e:
            print(f"DBSCAN failed: {e}")
        
        # Save clustering analysis
        plt.figure(figsize=(10, 6))
        plt.plot(k_values[:len(silhouette_scores)], silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('K-means Clustering: Silhouette Analysis')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'clustering_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        return clusters_info
    
    def _visualize_latent_space(self, latent_vectors: np.ndarray, clusters_info: dict, 
                              save_dir: str, original_images: np.ndarray = None):
        """Visualize latent space using dimensionality reduction techniques."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import umap
        import seaborn as sns
        
        print("Applying dimensionality reduction for visualization...")
        
        # Get clustering labels (use best clustering method)
        if 'kmeans' in clusters_info:
            labels = clusters_info['kmeans']['labels']
        else:
            labels = np.zeros(len(latent_vectors))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(latent_vectors)
        
        scatter = axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 0].set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        # 2. t-SNE
        if len(latent_vectors) > 5:
            perplexity = min(30, len(latent_vectors) // 4)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_result = tsne.fit_transform(latent_vectors)
            
            scatter = axes[0, 1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.7, s=20)
            axes[0, 1].set_title('t-SNE')
            axes[0, 1].set_xlabel('t-SNE 1')
            axes[0, 1].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')
        
        # 3. UMAP
        if len(latent_vectors) > 10:
            try:
                n_neighbors = min(15, len(latent_vectors) // 2)
                umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
                umap_result = umap_reducer.fit_transform(latent_vectors)
                
                scatter = axes[0, 2].scatter(umap_result[:, 0], umap_result[:, 1], 
                                           c=labels, cmap='tab10', alpha=0.7, s=20)
                axes[0, 2].set_title('UMAP')
                axes[0, 2].set_xlabel('UMAP 1')
                axes[0, 2].set_ylabel('UMAP 2')
                plt.colorbar(scatter, ax=axes[0, 2], label='Cluster')
            except Exception as e:
                axes[0, 2].text(0.5, 0.5, f'UMAP failed: {str(e)[:30]}...', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # 4. Latent dimensions correlation
        if self.latent_dim <= 20:
            corr_matrix = np.corrcoef(latent_vectors.T)
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Latent Dimensions Correlation')
            axes[1, 0].set_xlabel('Latent Dimension')
            axes[1, 0].set_ylabel('Latent Dimension')
            plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Latent space statistics
        axes[1, 1].hist(latent_vectors.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 1].set_title('Latent Space Value Distribution')
        axes[1, 1].set_xlabel('Latent Value')
        axes[1, 1].set_ylabel('Density')
        
        # 6. Cluster centers heatmap
        if 'kmeans' in clusters_info:
            centers = clusters_info['kmeans']['centers']
            im = axes[1, 2].imshow(centers, aspect='auto', cmap='viridis')
            axes[1, 2].set_title('Cluster Centers')
            axes[1, 2].set_xlabel('Latent Dimension')
            axes[1, 2].set_ylabel('Cluster')
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_space_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def _generate_cluster_representatives(self, latent_vectors: np.ndarray, clusters_info: dict, save_dir: str):
        """Generate representative skins for each cluster."""
        if 'kmeans' not in clusters_info:
            print("No clustering information available for generating representatives")
            return
        
        print("Generating cluster representative skins...")
        
        labels = clusters_info['kmeans']['labels']
        centers = clusters_info['kmeans']['centers']
        n_clusters = clusters_info['kmeans']['n_clusters']
        
        # Create subdirectory for cluster representatives
        cluster_dir = os.path.join(save_dir, 'cluster_representatives')
        os.makedirs(cluster_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, n_clusters, figsize=(3*n_clusters, 6))
        if n_clusters == 1:
            axes = axes.reshape(2, 1)
        
        with torch.no_grad():
            for cluster_id in range(n_clusters):
                # Generate skin from cluster center
                center_tensor = torch.tensor(centers[cluster_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                generated_center = self.model.decode(center_tensor).cpu().numpy().squeeze()
                generated_center = np.transpose(generated_center, (1, 2, 0))
                generated_center = np.clip(generated_center, 0, 1)
                
                # Find closest actual latent vector to center
                cluster_mask = labels == cluster_id
                cluster_vectors = latent_vectors[cluster_mask]
                if len(cluster_vectors) > 0:
                    distances = np.linalg.norm(cluster_vectors - centers[cluster_id], axis=1)
                    closest_idx = np.argmin(distances)
                    closest_vector = cluster_vectors[closest_idx]
                    
                    closest_tensor = torch.tensor(closest_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                    generated_closest = self.model.decode(closest_tensor).cpu().numpy().squeeze()
                    generated_closest = np.transpose(generated_closest, (1, 2, 0))
                    generated_closest = np.clip(generated_closest, 0, 1)
                else:
                    generated_closest = generated_center
                
                # Display
                axes[0, cluster_id].imshow(generated_center[:, :, :3])
                axes[0, cluster_id].set_title(f'Cluster {cluster_id} Center')
                axes[0, cluster_id].axis('off')
                
                axes[1, cluster_id].imshow(generated_closest[:, :, :3])
                axes[1, cluster_id].set_title(f'Closest to Center')
                axes[1, cluster_id].axis('off')
                
                # Save images
                center_rgba = (generated_center * 255).astype(np.uint8)
                closest_rgba = (generated_closest * 255).astype(np.uint8)
                
                Image.fromarray(center_rgba, 'RGBA').save(
                    os.path.join(cluster_dir, f'cluster_{cluster_id}_center.png'))
                Image.fromarray(closest_rgba, 'RGBA').save(
                    os.path.join(cluster_dir, f'cluster_{cluster_id}_closest.png'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cluster_representatives.png'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Cluster representatives saved to {cluster_dir}/")
        

def main():
    """Main function for VAE exploration utilities."""
    parser = argparse.ArgumentParser(description='Explore trained Minecraft Skin VAE')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['generate', 'interpolate', 'reconstruct', 'encode', 'analyze'],
                       default='generate', help='Exploration mode')
    parser.add_argument('--num_skins', type=int, default=8, help='Number of skins to generate')
    parser.add_argument('--skin_path', type=str, help='Path to skin file (for reconstruct/encode modes)')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset directory (for analyze mode)')
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
    
    elif args.mode == 'analyze':
        dataset_path = args.dataset_path or "/home/beto/MCSkins/skins/"
        print(f"Analyzing latent space from dataset: {dataset_path}")
        explorer.analyze_latent_space_from_dataset(dataset_path, max_samples=1000, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
