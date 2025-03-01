import numpy as np
import os
import pickle
from models import RBM
from utils import (load_binary_alphadigits,
                   display_binary_images,
                   display_weights,
                   plot_losses)

# Create directories for saving results if they don't exist
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

# Parameters - as specified in the project
n_hidden = 100
nb_epochs = 100  # As per project: 100 for RBM
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

# Determine original image dimensions (assuming square images)
# For alphadigits, this is typically 20x16 pixels
img_height = 20  # Update these values based on your actual dataset
img_width = 16
print(f"Original image dimensions: {img_height}x{img_width}")

# Display some random original samples
print("Displaying original samples:")
random_indices = np.random.choice(len(data), size=10, replace=False)
random_samples = data[random_indices]
display_binary_images(random_samples, n_cols=5, figsize=(10, 5), 
                     titles=[f"Sample {i}" for i in range(10)])

# Initialize and train RBM
print(f"Initializing RBM with {data.shape[1]} visible and {n_hidden} hidden units")
rbm = RBM(n_visible=data.shape[1], n_hidden=n_hidden)

print(f"Training RBM for {nb_epochs} epochs...")
rbm.fit(data, nb_epochs=nb_epochs, batch_size=batch_size, lr=learning_rate, k=k, verbose=True)

# Save the trained model
model_path = "results/models/rbm_alpha_digits.pkl"
print(f"Saving trained RBM model to {model_path}")
with open(model_path, 'wb') as f:
    pickle.dump(rbm, f)

# Generate samples
print("Generating samples from the trained RBM...")
samples = rbm.generate_samples(n_samples=10, gibbs_steps=1000)

# Display generated samples - reshape to original dimensions
print("Displaying generated samples:")
# Reshape samples to original dimensions if needed
samples_2d = [sample.reshape(img_height, img_width) for sample in samples]
display_binary_images(samples_2d, n_cols=5, figsize=(10, 5), 
                     titles=[f"Generated {i}" for i in range(10)],
                     save_path="results/plots/rbm_generated_samples.png")

# Reconstruct random samples
print("Reconstructing randomly selected samples...")
random_indices = np.random.choice(len(data), size=10, replace=False)
samples_to_reconstruct = data[random_indices]
reconstructions = rbm.reconstruct(samples_to_reconstruct)

# Reshape reconstructions to original dimensions
reconstructions_2d = [recon.reshape(img_height, img_width) for recon in reconstructions]
original_2d = [orig.reshape(img_height, img_width) for orig in samples_to_reconstruct]

# Display original and reconstructed samples side by side
print("Displaying original and reconstructed samples:")
all_images = original_2d + reconstructions_2d  # List concatenation
titles = [f"Original {i}" for i in range(10)] + [f"Reconstructed {i}" for i in range(10)]
display_binary_images(all_images, n_cols=10, figsize=(15, 5), 
                     titles=titles,
                     save_path="results/plots/rbm_reconstructions.png")

# Plot weights
print("Plotting RBM weights:")
display_weights(rbm, height=img_height, width=img_width, figsize=(10, 10), n_cols=10,
                save_path="results/plots/rbm_weights.png")

# Plot reconstruction error during training
print("Plotting reconstruction error:")
plot_losses(rbm.losses, 
           title='RBM Reconstruction Error',
           ylabel='Mean Squared Error',
           save_path="results/plots/rbm_reconstruction_error.png")

print("RBM training and visualization complete!")