import numpy as np
import os
import pickle
from models import DBN
from utils import (load_binary_alphadigits,
                   display_binary_images,
                   display_weights,
                   plot_losses)

# Create directories for saving results if they don't exist
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

# Parameters - adjusted to align with project specs
layer_sizes = [320, 200, 100]  # 20x16=320 input size
nb_epochs = 100  # As per project: 100 for RBM layers
batch_size = 10
learning_rate = 0.1  # As per project recommendation
k = 1  # CD-k steps
chars = list(range(10))  # Load digits 0-9

# Load data
print("Loading Binary AlphaDigits dataset...")
data = load_binary_alphadigits(chars=chars)

if data is None:
    print("Failed to load data. Exiting.")
    exit()

print(f"Loaded {data.shape[0]} samples with dimension {data.shape[1]}")

# Display some random original samples
print("Displaying original samples:")
random_indices = np.random.choice(len(data), size=10, replace=False)
random_samples = data[random_indices]
display_binary_images(random_samples, n_cols=5, figsize=(10, 5), 
                      titles=[f"Sample {i}" for i in range(10)])

# Initialize and train DBN
print(f"Initializing DBN with layers: {layer_sizes}")
dbn = DBN(layer_sizes)

print(f"Training DBN for {nb_epochs} epochs per layer...")
dbn.fit(data, nb_epochs=nb_epochs, batch_size=batch_size, lr=learning_rate, k=k, verbose=True)

# Save the trained model
model_path = "results/models/dbn_alpha_digits.pkl"
print(f"Saving trained DBN model to {model_path}")
with open(model_path, 'wb') as f:
    pickle.dump(dbn, f)

# Generate samples
print("Generating samples from the trained DBN...")
samples = dbn.predict(np.random.binomial(1, 0.5, (10, layer_sizes[0])))

# Convert generated samples to binary (black and white) instead of grayscale
samples = np.round(samples).astype(int)

# Display generated samples
print("Displaying generated samples:")
display_binary_images(samples, n_cols=5, figsize=(10, 5), titles=[f"Generated {i}" for i in range(10)], save_path="results/plots/dbn_generated_samples.png")

# Reconstruct random samples
print("Reconstructing randomly selected samples...")
random_indices = np.random.choice(len(data), size=10, replace=False)
samples_to_reconstruct = data[random_indices]
reconstructions = dbn.predict(samples_to_reconstruct)

# Convert reconstructions to binary (black and white) instead of grayscale
reconstructions = np.round(reconstructions).astype(int)

# Display original and reconstructed samples side by side
print("Displaying original and reconstructed samples:")
all_images = np.vstack([samples_to_reconstruct, reconstructions])
titles = [f"Original {i}" for i in range(10)] + [f"Reconstructed {i}" for i in range(10)]
display_binary_images(all_images, n_cols=10, figsize=(15, 5), titles=titles, save_path="results/plots/dbn_reconstructions.png")

# Plot pretraining errors
print("Plotting pretraining errors:")
if hasattr(dbn, 'pretrain_errors') and dbn.pretrain_errors:
    plot_losses(dbn.pretrain_errors, 
               title="Pretraining errors by layer",
               xlabel="Epoch",
               ylabel="Reconstruction Error",
               save_path="results/plots/dbn_pretraining_errors.png")

# Plot weights for each RBM layer in the DBN
print("Displaying DBN weights for each layer:")
for i, rbm in enumerate(dbn.rbms):
    print(f"Plotting weights for layer {i+1}/{len(dbn.rbms)}...")
    
    # For first layer: use original image dimensions
    if i == 0:
        height, width = 20, 16
    else:
        # For subsequent layers: calculate appropriate dimensions based on the previous layer size
        # Try to find a reasonable aspect ratio
        prev_layer_size = layer_sizes[i]
        # Find factors close to a square
        factors = []
        for j in range(1, int(np.sqrt(prev_layer_size)) + 1):
            if prev_layer_size % j == 0:
                factors.append((j, prev_layer_size // j))
        # Choose the factor pair with ratio closest to 1 (most square-like)
        height, width = min(factors, key=lambda x: abs(x[0]/x[1] - 1))
    
    display_weights(rbm, height=height, width=width, figsize=(10, 10), n_cols=10, 
                    save_path=f"results/plots/dbn_layer{i+1}_weights.png")

print("DBN training and visualization complete!")