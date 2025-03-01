import numpy as np
import os
import pickle
from models.dbn import DBN
from utils import load_binary_alphadigits, display_binary_images, plot_loss_curve

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

# Display generated samples
print("Displaying generated samples:")
display_binary_images(samples, n_cols=5, figsize=(10, 5), titles=[f"Generated {i}" for i in range(10)], save_path="results/plots/dbn_generated_samples.png")

# Reconstruct random samples
print("Reconstructing randomly selected samples...")
random_indices = np.random.choice(len(data), size=10, replace=False)
samples_to_reconstruct = data[random_indices]
reconstructions = dbn.predict(samples_to_reconstruct)

# Display original and reconstructed samples side by side
print("Displaying original and reconstructed samples:")
all_images = np.vstack([samples_to_reconstruct, reconstructions])
titles = [f"Original {i}" for i in range(10)] + [f"Reconstructed {i}" for i in range(10)]
display_binary_images(all_images, n_cols=10, figsize=(15, 5), titles=titles, save_path="results/plots/dbn_reconstructions.png")

# Plot pretraining errors
print("Plotting pretraining errors:")
# Assuming DBN class has a method to get pretraining errors
# If the plot_pretraining_errors method returns a figure:
if hasattr(dbn, 'plot_pretraining_errors'):
    if hasattr(dbn, 'pretraining_errors'):
        # Use our plot_loss_curve function instead
        for i, errors in enumerate(dbn.pretraining_errors):
            plot_loss_curve(
                errors,
                title=f'Layer {i+1} Pretraining Error',
                xlabel='Epoch',
                ylabel='Reconstruction Error',
                save_path=f"results/plots/dbn_layer{i+1}_pretraining_error.png"
            )
    else:
        # Fall back to the original method if needed
        try:
            fig_errors = dbn.plot_pretraining_errors()
            if fig_errors:  # Checking if the method returns a figure
                fig_errors.savefig("results/plots/dbn_pretraining_errors.png")
                print("Pretraining errors plot saved to results/plots/dbn_pretraining_errors.png")
        except Exception as e:
            print(f"Could not plot pretraining errors: {e}")

# Plot weights for each RBM layer in the DBN
print("Plotting DBN weights for each layer:")
for i, rbm in enumerate(dbn.rbms):
    print(f"Plotting weights for layer {i+1}/{len(dbn.rbms)}...")
    rbm.plot_weights(figsize=(10, 10), n_cols=10, 
                    save_path=f"results/plots/dbn_layer{i+1}_weights.png")

print("DBN training and visualization complete!")