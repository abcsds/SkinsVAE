# Minecraft Skins Variational Autoencoder (VAE)

This project implements a Convolutional Variational Autoencoder (VAE) to learn a statistical latent space representation of Minecraft skins. The VAE learns to encode 64x64 RGBA skin images into a lower-dimensional latent space and decode them back to the original format.

## Features

- **Convolutional Architecture**: Uses convolutional layers for both encoder and decoder to preserve spatial relationships
- **Configurable Latent Space**: Default 20D latent space for rich representation with clustering analysis for visualization
- **Unsupervised Clustering**: K-means and DBSCAN clustering of latent representations
- **Dimensionality Reduction**: t-SNE, UMAP, and PCA for 2D visualization of high-dimensional latent space
- **GPU Acceleration**: Optimized for NVIDIA GPUs (tested on GTX 1080 Ti)
- **TensorBoard Integration**: Comprehensive logging and visualization
- **Checkpointing**: Save and resume training with model checkpoints
- **Batch Training**: Efficient batch processing with data loading
- **Beta-VAE Support**: Configurable beta parameter for disentangled representations

## Architecture

### Model Architecture

The VAE consists of:

#### Encoder (Convolutional)
```
Input: 64x64x4 (RGBA)
├── Conv2d(4→32, k=4, s=2, p=1) + ReLU + BatchNorm → 32x32x32
├── Conv2d(32→64, k=4, s=2, p=1) + ReLU + BatchNorm → 16x16x64
├── Conv2d(64→128, k=4, s=2, p=1) + ReLU + BatchNorm → 8x8x128
├── Conv2d(128→256, k=4, s=2, p=1) + ReLU + BatchNorm → 4x4x256
├── Flatten → 4096
├── Linear(4096 → latent_dim) → μ (mean)
└── Linear(4096 → latent_dim) → σ (log variance)
```

#### Decoder (Transposed Convolutional)
```
Input: latent_dim
├── Linear(latent_dim → 4096) → 4x4x256
├── ConvTranspose2d(256→128, k=4, s=2, p=1) + ReLU + BatchNorm → 8x8x128
├── ConvTranspose2d(128→64, k=4, s=2, p=1) + ReLU + BatchNorm → 16x16x64
├── ConvTranspose2d(64→32, k=4, s=2, p=1) + ReLU + BatchNorm → 32x32x32
└── ConvTranspose2d(32→4, k=4, s=2, p=1) + Sigmoid → 64x64x4
```

### Loss Function

The VAE loss combines reconstruction loss and KL divergence:

```
L = L_recon + β × L_KL

where:
- L_recon = MSE(x, x_reconstructed)
- L_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
- β = regularization parameter (default: 1.0)
```

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/beto/skinsVAE
   ```

2. **Set up virtual environment and install dependencies**:
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   
   # Activate virtual environment
   source .venv/bin/activate
   
   # Upgrade pip and install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Create necessary directories**:
   ```bash
   mkdir -p checkpoints runs exploration_output
   ```

4. **Verify installation**:
   ```bash
   # Activate virtual environment (if not already active)
   source .venv/bin/activate
   
   # Test PyTorch installation
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('Installation successful!')"
   ```

**Note**: This project is currently configured with CPU-only PyTorch for compatibility. For GPU acceleration, you may need to install a CUDA-compatible version based on your GPU specifications.

## Usage

**Remember to activate the virtual environment before running any commands:**
```bash
source .venv/bin/activate
```

### Basic Training

Train the VAE with default parameters (3D latent space):

```bash
python minecraft_skin_vae.py --data_dir /home/beto/MCSkins/skins/
```

### Advanced Training Options

```bash
python minecraft_skin_vae.py \
    --data_dir /home/beto/MCSkins/skins/ \
    --latent_dim 3 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --beta 1.0 \
    --checkpoint_dir checkpoints \
    --save_freq 10
```

### Resume Training

Resume training from a checkpoint:

```bash
python minecraft_skin_vae.py \
    --data_dir /home/beto/MCSkins/skins/ \
    --resume checkpoints/vae_epoch_100.pth
```

### Parameters

- `--data_dir`: Directory containing PNG skin files (default: `/home/beto/MCSkins/skins/`)
- `--latent_dim`: Latent space dimensionality (default: 3)
- `--batch_size`: Training batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for Adam optimizer (default: 0.001)
- `--beta`: Beta parameter for beta-VAE (default: 1.0)
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--resume`: Path to checkpoint file to resume from
- `--save_freq`: Save checkpoint every N epochs (default: 10)

## TensorBoard Visualization

The training process logs comprehensive metrics and visualizations to TensorBoard:

### 1. Start TensorBoard

After starting training, TensorBoard logs are saved in the `runs/` directory. Start TensorBoard:

```bash
tensorboard --logdir=runs
```

Then open your browser to `http://localhost:6006`

### 2. Available Visualizations

#### Scalar Metrics
- **Loss/Batch_Total**: Total loss per batch
- **Loss/Batch_Reconstruction**: Reconstruction loss per batch  
- **Loss/Batch_KL**: KL divergence loss per batch
- **Loss/Epoch_Total**: Average total loss per epoch
- **Loss/Epoch_Reconstruction**: Average reconstruction loss per epoch
- **Loss/Epoch_KL**: Average KL divergence loss per epoch

#### Image Visualizations
- **Reconstructions**: Side-by-side comparison of original and reconstructed skins (logged every 5 epochs)

#### Latent Space Visualizations
- **Latent_Space_3D**: 3D scatter plot of latent embeddings (for 3D latent space)
- **Latent_Space_2D**: 2D scatter plot of latent embeddings (for 2D latent space)

### 3. Monitoring Training

Key metrics to monitor:

1. **Total Loss**: Should decrease steadily
2. **Reconstruction Loss**: Measures how well the model reconstructs images
3. **KL Loss**: Measures how well the latent space follows a normal distribution
4. **Reconstructions**: Visual quality should improve over epochs
5. **Latent Space**: Should show meaningful clustering and smooth transitions

## Latent Space Analysis

The project includes advanced latent space analysis with:

#### Clustering Algorithms
- **K-means**: Automatic optimal cluster detection using silhouette analysis
- **DBSCAN**: Density-based clustering for irregular cluster shapes

#### Dimensionality Reduction
- **PCA**: Principal Component Analysis with explained variance
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for local structure
- **UMAP**: Uniform Manifold Approximation and Projection for global structure

#### Visualization Features
- 2D scatter plots with cluster coloring
- Correlation matrices of latent dimensions
- Cluster center heatmaps
- Representative skin generation for each cluster

## File Structure

```
skinsVAE/
├── minecraft_skin_vae.py      # Main training script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── checkpoints/              # Model checkpoints (created during training)
│   ├── vae_epoch_10.pth
│   ├── vae_epoch_20.pth
│   └── ...
└── runs/                     # TensorBoard logs (created during training)
    └── skin_vae_YYYYMMDD_HHMMSS/
        ├── events.out.tfevents.*
        └── ...
```

## Model Checkpoints

Checkpoints are automatically saved every 10 epochs (configurable) and contain:

- Model state dictionary
- Optimizer state dictionary  
- Current epoch
- Current loss
- Latent dimension configuration

### Loading a Trained Model

To load a trained model for inference:

```python
import torch
from minecraft_skin_vae import ConvVAE

# Load checkpoint
checkpoint = torch.load('checkpoints/vae_epoch_100.pth')
latent_dim = checkpoint['latent_dim']

# Create and load model
model = ConvVAE(latent_dim=latent_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    # Encode an image
    mu, logvar = model.encode(image_tensor)
    
    # Generate new images
    z = torch.randn(1, latent_dim)  # Sample from prior
    generated_image = model.decode(z)
```

## Tips for Training

### 1. Latent Dimension Selection
- **3D**: Good for visualization and initial experiments
- **8-16D**: Better reconstruction quality
- **32-128D**: High-quality reconstructions but less interpretable

### 2. Beta Parameter Tuning
- **β = 1.0**: Standard VAE
- **β < 1.0**: Emphasizes reconstruction quality
- **β > 1.0**: Emphasizes disentangled representations (beta-VAE)

### 3. Monitoring Training
- Watch for mode collapse (KL loss goes to 0)
- Ensure reconstruction loss decreases steadily
- Check visual quality in TensorBoard reconstructions

### 4. Hardware Considerations
- GTX 1080 Ti (11GB): Can handle batch sizes up to 64-128
- Reduce batch size if running out of GPU memory
- Use `num_workers=4` in DataLoader for faster data loading

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size`
2. **No PNG files found**: Check the `data_dir` path
3. **Poor reconstruction quality**: Try lowering beta or increasing latent dimension
4. **Training too slow**: Ensure CUDA is available and being used

### Performance Optimization

- Use mixed precision training for faster training on modern GPUs
- Increase `num_workers` in DataLoader if CPU allows
- Use SSD storage for faster data loading
- Monitor GPU utilization with `nvidia-smi`

## Future Enhancements

- [ ] Add data augmentation (rotation, color jittering)
- [ ] Implement conditional VAE for controlled generation
- [ ] Add interpolation utilities for latent space exploration
- [ ] Support for different skin formats (e.g., slim vs classic)
- [ ] Web interface for interactive exploration
