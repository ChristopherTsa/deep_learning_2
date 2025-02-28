import numpy as np
import matplotlib.pyplot as plt
from models.rbm import RBM
from utils.data_utils import load_binary_alphadigits, display_binary_images

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

# Display some original samples
print("Displaying original samples:")
display_binary_images(data[:10], n_cols=5, figsize=(10, 5), titles=[f"Sample {i}" for i in range(10)])

# Initialize and train RBM
print(f"Initializing RBM with {data.shape[1]} visible and {n_hidden} hidden units")
rbm = RBM(n_visible=data.shape[1], n_hidden=n_hidden)

print(f"Training RBM for {nb_epochs} epochs...")
rbm.fit(data, nb_epochs=nb_epochs, batch_size=batch_size, lr=learning_rate, k=k, verbose=True)

# Generate samples
print("Generating samples from the trained RBM...")
samples = rbm.generate_samples(n_samples=10, gibbs_steps=1000)

# Display generated samples
print("Displaying generated samples:")
display_binary_images(samples, n_cols=5, figsize=(10, 5), titles=[f"Generated {i}" for i in range(10)])

# Reconstruct original samples
print("Reconstructing original samples...")
reconstructions = rbm.reconstruct(data[:10])

# Display original and reconstructed samples side by side
print("Displaying original and reconstructed samples:")
all_images = np.vstack([data[:10], reconstructions])
titles = [f"Original {i}" for i in range(10)] + [f"Reconstructed {i}" for i in range(10)]
display_binary_images(all_images, n_cols=10, figsize=(15, 5), titles=titles)

# Plot weights
print("Plotting RBM weights:")
rbm.plot_weights(figsize=(10, 10), n_cols=10)

# Plot reconstruction error during training
plt.figure(figsize=(10, 6))
plt.plot(rbm.losses)
plt.title('RBM Reconstruction Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

print("RBM training and visualization complete!")