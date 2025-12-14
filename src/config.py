# Autoencoder layer boyutları
LAYER_SIZES = [784, 400, 200, 100, 50, 25, 6]  # Encoder layers
CODE_SIZE = 6  # Code layer size

# RBM pretraining parameters
RBM_K = 1  # Gibbs sampling step
RBM_LR = 0.01  # Learning rate

# Fine-tuning parameters
FINE_TUNE_LR = 0.001
FINE_TUNE_EPOCHS = 50

# General parameters
DEVICE = ‘cuda’  # or 'cpu'
