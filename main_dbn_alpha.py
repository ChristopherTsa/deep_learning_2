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
layer_sizes = [320, 200, 200]  # 20x16=320 input size
nb_epochs = 1000
batch_size = 10
learning_rate = 0.01
k = 1  # CD-k steps
chars = [10, 11, 12, 13, 14] # Characters to load

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
dbn.fit(data,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        k=k,
        verbose=True)

# Save the trained model
model_path = "results/models/dbn_alpha_digits.pkl"
print(f"Saving trained DBN model to {model_path}")
with open(model_path, 'wb') as f:
    pickle.dump(dbn, f)

# Generate samples
print("Generating samples from the trained DBN...")
samples = dbn.generate_samples(n_samples=25, gibbs_steps=200)

# Display generated samples
print("Displaying generated samples:")
display_binary_images(samples, n_cols=5, figsize=(10, 5), titles=[f"Generated {i}" for i in range(10)], save_path="results/plots/dbn_generated_samples.png")

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

# ========= Varying the hidden dimension =========
print("\n========= Experiment: Varying the hidden dimension =========")

# List of hidden dimensions to try
hidden_dims = [100, 200, 400]
errors_by_dim = {}

print(f"Running experiment with hidden dimensions: {hidden_dims}")
print(f"Using characters: {chars}")

for dim in hidden_dims:
    print(f"\nTraining DBN with {dim} units in hidden layers...")
    
    # Initialize DBN with current hidden dimension
    current_layers = [320, dim, dim]
    
    print(f"Initializing DBN with layers: {current_layers}")
    dbn_dim = DBN(current_layers)
    
    # Train the model
    dbn_dim.fit(data,
              nb_epochs=nb_epochs,
              batch_size=batch_size,
              lr=learning_rate,
              k=k,
              verbose=True)
    
    # Save the model
    model_path = f"results/models/dbn_alpha_h{dim}.pkl"
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(dbn_dim, f)
    
    # Store pretraining errors if available
    if hasattr(dbn_dim, 'pretrain_errors') and dbn_dim.pretrain_errors:
        errors_by_dim[dim] = dbn_dim.pretrain_errors
    
    # Generate and display samples
    samples = dbn_dim.generate_samples(n_samples=25, gibbs_steps=200)
    display_binary_images(samples, n_cols=5, figsize=(10, 5),
                        titles=[f"Hidden={dim}, {i}" for i in range(10)],
                        save_path=f"results/plots/dbn_h{dim}_samples.png")
    
    # Display weights for first layer
    display_weights(dbn_dim.rbms[0], height=20, width=16, figsize=(10, 10), n_cols=10,
                   save_path=f"results/plots/dbn_h{dim}_weights_layer1.png")

# Plot comparative errors if available
if errors_by_dim:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for dim, errors in errors_by_dim.items():
        plt.plot(errors, label=f'Hidden Units: {dim}')
    plt.xlabel('Epoch')
    plt.ylabel('Pretraining Error')
    plt.title('DBN Pretraining Error for Different Hidden Dimensions')
    plt.legend()
    plt.savefig("results/plots/dbn_hidden_dim_comparison.png")
    plt.close()

# ========= Varying the number of hidden layers =========
print("\n========= Experiment: Varying the number of hidden layers =========")

# Define different layer architectures to try
layer_architectures = [
    [320, 200],                  # 1 hidden layer
    [320, 200, 200],             # 2 hidden layers
    [320, 200, 200, 200],        # 3 hidden layers
    [320, 200, 200, 200, 200]    # 4 hidden layers
]

errors_by_arch = {}

for layers in layer_architectures:
    layer_name = '_'.join(map(str, layers[1:]))  # Skip input layer in the name
    print(f"\nTraining DBN with architecture: {layers}")
    
    # Initialize DBN with current architecture
    dbn_layers = DBN(layers)
    
    # Train the model
    dbn_layers.fit(data,
                 nb_epochs=nb_epochs,
                 batch_size=batch_size,
                 lr=learning_rate,
                 k=k,
                 verbose=True)
    
    # Save the model
    model_path = f"results/models/dbn_alpha_layers_{layer_name}.pkl"
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(dbn_layers, f)
    
    # Store pretraining errors if available
    if hasattr(dbn_layers, 'pretrain_errors') and dbn_layers.pretrain_errors:
        errors_by_arch[layer_name] = dbn_layers.pretrain_errors
    
    # Generate and display samples
    samples = dbn_layers.generate_samples(n_samples=25, gibbs_steps=200)
    display_binary_images(samples, n_cols=5, figsize=(10, 5),
                        titles=[f"Layers={layer_name}, {i}" for i in range(10)],
                        save_path=f"results/plots/dbn_layers_{layer_name}_samples.png")
    
    # Display weights for first layer
    display_weights(dbn_layers.rbms[0], height=20, width=16, figsize=(10, 10), n_cols=10,
                   save_path=f"results/plots/dbn_layers_{layer_name}_weights_layer1.png")

# Plot comparative errors if available
if errors_by_arch:
    plt.figure(figsize=(12, 6))
    for arch_name, errors in errors_by_arch.items():
        plt.plot(errors, label=f'Architecture: {arch_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Pretraining Error')
    plt.title('DBN Pretraining Error for Different Layer Architectures')
    plt.legend()
    plt.savefig("results/plots/dbn_layer_arch_comparison.png")
    plt.close()

# ========= Varying the number of characters =========
print("\n========= Experiment: Varying the number of characters =========")

# Define different character sets to try
char_sets = [
    [10, 11],      # Two characters
    [10, 11, 12, 13, 14],  # Five characters
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # Ten characters
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # Seventeen characters
]

errors_by_chars = {}

for char_set in char_sets:
    char_name = '_'.join(map(str, char_set))
    print(f"\nTraining DBN with characters: {char_set}")
    
    # Load data for current character set
    data_subset = load_binary_alphadigits(chars=char_set)
    print(f"Loaded {data_subset.shape[0]} samples")
    
    # Display some samples
    random_indices = np.random.choice(len(data_subset), size=10, replace=False)
    display_binary_images(data_subset[random_indices], n_cols=5, figsize=(10, 5),
                        titles=[f"Char {char_name} - {i}" for i in range(10)],
                        save_path=f"results/plots/dbn_orig_samples_chars_{char_name}.png")
    
    # Initialize and train DBN with default layer architecture
    dbn_chars = DBN(layer_sizes)
    
    # Train the model
    dbn_chars.fit(data_subset,
                nb_epochs=nb_epochs,
                batch_size=batch_size,
                lr=learning_rate,
                k=k,
                verbose=True)
    
    # Save the model
    model_path = f"results/models/dbn_alpha_chars_{char_name}.pkl"
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(dbn_chars, f)
    
    # Store pretraining errors if available
    if hasattr(dbn_chars, 'pretrain_errors') and dbn_chars.pretrain_errors:
        errors_by_chars[char_name] = dbn_chars.pretrain_errors
    
    # Generate and display samples
    samples = dbn_chars.generate_samples(n_samples=25, gibbs_steps=200)
    display_binary_images(samples, n_cols=5, figsize=(10, 5),
                        titles=[f"Chars={char_name}, {i}" for i in range(10)],
                        save_path=f"results/plots/dbn_chars_{char_name}_samples.png")
    
    # Display weights for first layer
    display_weights(dbn_chars.rbms[0], height=20, width=16, figsize=(10, 10), n_cols=10,
                   save_path=f"results/plots/dbn_chars_{char_name}_weights_layer1.png")

# Plot comparative errors if available
if errors_by_chars:
    plt.figure(figsize=(12, 6))
    for char_name, errors in errors_by_chars.items():
        plt.plot(errors, label=f'Characters: {char_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Pretraining Error')
    plt.title('DBN Pretraining Error for Different Character Sets')
    plt.legend()
    plt.savefig("results/plots/dbn_char_sets_comparison.png")
    plt.close()

print("All DBN experiments completed!")