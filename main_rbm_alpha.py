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
nb_epochs = 1000
batch_size = 10
learning_rate = 0.01
k = 1  # CD-k steps
chars = [10, 11, 12] # Characters to load

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

# Initialize and train RBM
print(f"Initializing RBM with {data.shape[1]} visible and {n_hidden} hidden units")
rbm = RBM(n_visible=data.shape[1], n_hidden=n_hidden)

print(f"Training RBM for {nb_epochs} epochs...")
rbm.fit(data,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        k=k,
        verbose=True)

# Save the trained model
model_path = "results/models/rbm_alpha_digits.pkl"
print(f"Saving trained RBM model to {model_path}")
with open(model_path, 'wb') as f:
    pickle.dump(rbm, f)

# Generate samples
print("Generating samples from the trained RBM...")
samples = rbm.generate_samples(n_samples=25, gibbs_steps=200)

# Display generated samples - reshape to original dimensions
print("Displaying generated samples:")
# Reshape samples to original dimensions if needed
display_binary_images(samples, n_cols=5, figsize=(10, 5), 
                     titles=[f"Generated {i}" for i in range(10)],
                     save_path="results/plots/rbm_generated_samples.png")

# Plot reconstruction error during training
print("Plotting reconstruction error:")
plot_losses(rbm.errors, 
           title='RBM Reconstruction Error',
           ylabel='Mean Squared Error',
           save_path="results/plots/rbm_reconstruction_error.png")

# Plot weights
print("Displaying RBM weights:")
display_weights(rbm, height=20, width=16, figsize=(10, 10), n_cols=10,
                save_path="results/plots/rbm_weights.png")

print("RBM training and visualization complete!")

# ========= Varying the hidden dimension =========
print("\n========= Experiment: Varying the hidden dimension =========")

# List of hidden dimensions to try
hidden_dims = [50, 100, 200, 400]
errors_by_dim = {}

print(f"Running experiment with hidden dimensions: {hidden_dims}")
print(f"Using characters: {chars}")

for dim in hidden_dims:
    print(f"\nTraining RBM with {dim} hidden units...")
    
    # Initialize RBM with current hidden dimension
    rbm_dim = RBM(n_visible=data.shape[1], n_hidden=dim)
    
    # Train the model
    rbm_dim.fit(data,
             nb_epochs=nb_epochs,
             batch_size=batch_size,
             lr=learning_rate,
             k=k,
             verbose=True)
    
    # Save the model
    model_path = f"results/models/rbm_alpha_h{dim}.pkl"
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(rbm_dim, f)
    
    # Store reconstruction errors
    errors_by_dim[dim] = rbm_dim.errors
    
    # Generate and display samples
    samples = rbm_dim.generate_samples(n_samples=25, gibbs_steps=200)
    display_binary_images(samples, n_cols=5, figsize=(10, 5),
                        titles=[f"Hidden={dim}, {i}" for i in range(10)],
                        save_path=f"results/plots/rbm_h{dim}_samples.png")
    
    # Display weights
    display_weights(rbm_dim, height=20, width=16, figsize=(10, 10), n_cols=10,
                   save_path=f"results/plots/rbm_h{dim}_weights.png")

# Plot comparative reconstruction errors
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for dim, errors in errors_by_dim.items():
    plt.plot(errors, label=f'Hidden Units: {dim}')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error')
plt.title('RBM Reconstruction Error for Different Hidden Dimensions')
plt.legend()
plt.savefig("results/plots/rbm_hidden_dim_comparison.png")
plt.close()

# ========= Varying the number of characters =========
print("\n========= Experiment: Varying the number of characters =========")

# Define different character sets to try
char_sets = [
    [10, 11],      # Two characters
    [10, 11, 12],  # Three characters
    [10, 11, 12, 13, 14],  # Five characters
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Ten characters
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  # Twenty characters
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]  # Twenty-six characters
]

errors_by_chars = {}

for char_set in char_sets:
    char_name = '_'.join(map(str, char_set))
    print(f"\nTraining RBM with characters: {char_set}")
    
    # Load data for current character set
    data_subset = load_binary_alphadigits(chars=char_set)
    print(f"Loaded {data_subset.shape[0]} samples")
    
    # Display some samples
    random_indices = np.random.choice(len(data_subset), size=10, replace=False)
    display_binary_images(data_subset[random_indices], n_cols=5, figsize=(10, 5),
                        titles=[f"Char {char_set} - {i}" for i in range(10)],
                        save_path=f"results/plots/orig_samples_chars_{char_name}.png")
    
    # Initialize and train RBM
    rbm_chars = RBM(n_visible=data_subset.shape[1], n_hidden=n_hidden)
    
    # Train the model
    rbm_chars.fit(data_subset,
                nb_epochs=nb_epochs,
                batch_size=batch_size,
                lr=learning_rate,
                k=k,
                verbose=True)
    
    # Save the model
    model_path = f"results/models/rbm_alpha_chars_{char_name}.pkl"
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(rbm_chars, f)
    
    # Store reconstruction errors
    errors_by_chars[char_name] = rbm_chars.errors
    
    # Generate and display samples
    samples = rbm_chars.generate_samples(n_samples=25, gibbs_steps=200)
    display_binary_images(samples, n_cols=5, figsize=(10, 5),
                        titles=[f"Chars={char_name}, {i}" for i in range(10)],
                        save_path=f"results/plots/rbm_chars_{char_name}_samples.png")
    
    # Display weights
    display_weights(rbm_chars, height=20, width=16, figsize=(10, 10), n_cols=10,
                   save_path=f"results/plots/rbm_chars_{char_name}_weights.png")

# Plot comparative reconstruction errors for different character sets
plt.figure(figsize=(12, 6))
for char_name, errors in errors_by_chars.items():
    plt.plot(errors, label=f'Characters: {char_name}')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error')
plt.title('RBM Reconstruction Error for Different Character Sets')
plt.legend()
plt.savefig("results/plots/rbm_char_sets_comparison.png")
plt.close()

print("All experiments completed!")