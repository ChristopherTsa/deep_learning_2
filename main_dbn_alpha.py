import numpy as np
from models.dbn import DBN
from utils import load_binary_alphadigits, display_binary_images

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

# Display some original samples
print("Displaying original samples:")
display_binary_images(data[:10], n_cols=5, figsize=(10, 5), titles=[f"Sample {i}" for i in range(10)])

# Initialize and train DBN
print(f"Initializing DBN with layers: {layer_sizes}")
dbn = DBN(layer_sizes)

print(f"Training DBN for {nb_epochs} epochs per layer...")
dbn.fit(data, nb_epochs=nb_epochs, batch_size=batch_size, lr=learning_rate, k=k, verbose=True)

# Generate samples
print("Generating samples from the trained DBN...")
samples = dbn.predict(np.random.binomial(1, 0.5, (10, layer_sizes[0])))

# Display generated samples
print("Displaying generated samples:")
display_binary_images(samples, n_cols=5, figsize=(10, 5), titles=[f"Generated {i}" for i in range(10)])

# Reconstruct original samples
print("Reconstructing original samples...")
reconstructions = dbn.predict(data[:10])

# Display original and reconstructed samples side by side
print("Displaying original and reconstructed samples:")
all_images = np.vstack([data[:10], reconstructions])
titles = [f"Original {i}" for i in range(10)] + [f"Reconstructed {i}" for i in range(10)]
display_binary_images(all_images, n_cols=10, figsize=(15, 5), titles=titles)

# Plot pretraining errors
print("Plotting pretraining errors:")
dbn.plot_pretraining_errors()

print("DBN training and visualization complete!")