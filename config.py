"""
Configuration file for Minecraft Skins VAE
"""

# Data configuration
DATA_DIR = "/home/beto/MCSkins/skins/"
IMAGE_SIZE = 64
NUM_CHANNELS = 4  # RGBA

# Model configuration
LATENT_DIM = 3  # 3D for visualization, can be changed for experiments
BETA = 0.5  # Beta-VAE parameter

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Checkpoint configuration
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQ = 100  # Save every N epochs
RESUME_FROM = None  # Path to checkpoint to resume from

# Logging configuration
LOG_DIR = "runs"
LOG_IMAGES = True
LOG_FREQ = 5  # Log images every N epochs

# Device configuration (will be set dynamically in main script)
NUM_WORKERS = 4  # For data loading

# Model parameters
ENCODER_CHANNELS = [4, 32, 64, 128, 256]
DECODER_CHANNELS = [256, 128, 64, 32, 4]
