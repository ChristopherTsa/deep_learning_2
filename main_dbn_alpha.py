import numpy as np
import os
import pickle
from models import DBN
from utils import (load_binary_alphadigits,
                   display_binary_images,
                   display_weights,
                   plot_losses,
                   create_data_splits)

# Create directories for saving results if they don't exist
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

def train_dbn(data, val_data=None, layer_sizes=None, nb_epochs=100, batch_size=10, 
            learning_rate=0.1, k=1, model_name="dbn", verbose=True, save_model=True,
            save_samples=True, save_weights=True, plot=True):
    """
    Train a DBN model and optionally visualize results.
    
    Parameters:
    -----------
    data: ndarray
        Training data of shape (n_samples, n_features)
    val_data: ndarray, optional
        Validation data for monitoring reconstruction loss on unseen data
    layer_sizes: list
        List of layer sizes (incl. visible layer)
    nb_epochs: int
        Number of training epochs
    batch_size: int
        Mini-batch size
    learning_rate: float
        Learning rate for training
    k: int
        Number of Gibbs sampling steps (CD-k)
    model_name: str
        Name identifier for saved files
    verbose: bool
        Whether to print progress during training
    save_model: bool
        Whether to save the trained model
    save_samples: bool
        Whether to generate and save samples
    save_weights: bool
        Whether to visualize and save weights
        
    Returns:
    --------
    dbn: DBN
        Trained DBN model
    """
    # Initialize and train DBN
    if verbose:
        print(f"Initializing DBN with layers: {layer_sizes}")
    
    dbn = DBN(layer_sizes)

    # Train the model
    if verbose:
        print(f"Training DBN for {nb_epochs} epochs per layer...")
    
    dbn.fit(data,
            validation_data=val_data,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            k=k,
            verbose=verbose)
    
    # Save the trained model
    if save_model:
        model_path = f"results/models/{model_name}.pkl"
        if verbose:
            print(f"Saving trained DBN model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(dbn, f)

    # Generate and display samples
    if save_samples:
        if verbose:
            print("Generating samples from the trained DBN...")
        samples = dbn.generate_samples(n_samples=25, gibbs_steps=200)
        
        if verbose:
            print("Displaying generated samples:")
        display_binary_images(samples, n_cols=5, figsize=(10, 5),
                             save_path=f"results/plots/{model_name}_samples.png")
    
    if plot:
        # Plot pretraining losses
        if verbose:
            print("Plotting pretraining losses:")
        if hasattr(dbn, 'pretrain_losses') and dbn.pretrain_losses:
            plot_losses(dbn.pretrain_losses, 
                       title="Pretraining losses by layer",
                       xlabel="Epoch",
                       ylabel="Reconstruction Loss",
                       save_path=f"results/plots/{model_name}_pretraining_losses.png")

    # Plot weights for each RBM layer in the DBN
    if save_weights:
        if verbose:
            print("Displaying DBN weights for each layer:")
        for i, rbm in enumerate(dbn.rbms):
            if verbose:
                print(f"Plotting weights for layer {i+1}/{len(dbn.rbms)}...")
            
            # For first layer: use original image dimensions
            if i == 0:
                height, width = 20, 16
            else:
                # For subsequent layers: calculate appropriate dimensions
                prev_layer_size = layer_sizes[i]
                factors = []
                for j in range(1, int(np.sqrt(prev_layer_size)) + 1):
                    if prev_layer_size % j == 0:
                        factors.append((j, prev_layer_size // j))
                height, width = min(factors, key=lambda x: abs(x[0]/x[1] - 1))
            
            display_weights(rbm, height=height, width=width, figsize=(10, 10), n_cols=10, 
                           save_path=f"results/plots/{model_name}_layer{i+1}_weights.png")
    
    return dbn

def experiment_hidden_dimensions(train_data, val_data=None, hidden_dims=[50, 100, 200, 400], 
                               nb_epochs=100, batch_size=10, learning_rate=0.1, k=1, chars=None):
    """
    Experiment with varying hidden dimensions in DBN.
    
    Parameters:
    -----------
    train_data: ndarray
        Training data
    val_data: ndarray, optional
        Validation data
    hidden_dims: list
        List of hidden dimensions to try
    nb_epochs, batch_size, learning_rate, k: 
        Training parameters
    chars: list
        List of characters used (for logging)
        
    Returns:
    --------
    losses_by_dim: dict
        Dictionary mapping hidden dimensions to reconstruction losses
    best_dim: int
        Hidden dimension with lowest pretraining loss
    """
    print(f"\n========= Experiment: Varying the hidden dimension =========")
    losses_by_dim = {}
    best_loss = float('inf')
    best_dim = None

    print(f"Running experiment with hidden dimensions: {hidden_dims}")
    if chars:
        print(f"Using characters: {chars}")

    for dim in hidden_dims:
        print(f"\nTraining DBN with {dim} units in hidden layers...")
        
        # Initialize DBN with current hidden dimension
        current_layers = [train_data.shape[1], dim, dim]
        
        # Train DBN with current hidden dimension
        dbn_dim = train_dbn(
            data=train_data,
            val_data=val_data,
            layer_sizes=current_layers,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            k=k,
            model_name=f"dbn_alpha_h{dim}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store pretraining losses if available
        if hasattr(dbn_dim, 'pretrain_losses') and dbn_dim.pretrain_losses:
            final_loss = dbn_dim.pretrain_losses[-1][-1]
            losses_by_dim[dim] = dbn_dim.pretrain_losses
            
            # Check if this is the best model based on final loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_dim = dim

    print(f"Best hidden dimension: {best_dim} with pretraining loss: {best_loss:.6f}")
    return losses_by_dim, best_dim

def experiment_number_of_layers(train_data, val_data=None, num_layers=[1, 2, 3, 4], hidden_size=100,
                             nb_epochs=100, batch_size=10, learning_rate=0.1, k=1, chars=None):
    """
    Experiment with varying the number of layers in DBN.
    
    Parameters:
    -----------
    train_data: ndarray
        Training data
    val_data: ndarray, optional
        Validation data
    num_layers: list
        List of number of hidden layers to try
    hidden_size: int
        Size of each hidden layer
    nb_epochs, batch_size, learning_rate, k: 
        Training parameters
    chars: list
        List of characters used (for logging)
        
    Returns:
    --------
    losses_by_layers: dict
        Dictionary mapping number of layers to reconstruction losses
    best_num_layers: int
        Number of layers with lowest pretraining loss
    """
    print("\n========= Experiment: Varying the number of hidden layers =========")
    losses_by_layers = {}
    best_loss = float('inf')
    best_num_layers = None

    print(f"Running experiment with number of layers: {num_layers}")
    if chars:
        print(f"Using characters: {chars}")

    for n_layers in num_layers:
        print(f"\nTraining DBN with {n_layers} hidden layers...")
        
        # Create layer architecture with n_layers hidden layers of size hidden_size
        layer_architecture = [train_data.shape[1]] + [hidden_size] * n_layers
        layer_name = '_'.join(map(str, layer_architecture[1:]))
        
        # Train DBN with current layer architecture
        dbn_layers = train_dbn(
            data=train_data,
            val_data=val_data,
            layer_sizes=layer_architecture,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            k=k,
            model_name=f"dbn_alpha_layers_{layer_name}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store pretraining losses if available
        if hasattr(dbn_layers, 'pretrain_losses') and dbn_layers.pretrain_losses:
            final_loss = dbn_layers.pretrain_losses[-1][-1]
            losses_by_layers[n_layers] = dbn_layers.pretrain_losses
            
            # Check if this is the best model based on final loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_num_layers = n_layers

    print(f"Best number of layers: {best_num_layers} with pretraining loss: {best_loss:.6f}")
    return losses_by_layers, best_num_layers

def experiment_characters(char_sets, layer_sizes=None, nb_epochs=100, batch_size=10, 
                        learning_rate=0.1, k=1):
    """
    Experiment with varying character sets in DBN.
    
    Parameters:
    -----------
    char_sets: dict
        Dictionary mapping character set names to lists of characters
    layer_sizes, nb_epochs, batch_size, learning_rate, k: 
        Training parameters
        
    Returns:
    --------
    losses_by_chars: dict
        Dictionary mapping character sets to pretraining losses
    """
    print("\n========= Experiment: Varying the number of characters =========")
    losses_by_chars = {}

    for char_name, char_set in char_sets.items():
        print(f"\nTraining DBN with characters: {char_name}")
        
        # Load data for current character set
        data_subset = load_binary_alphadigits(chars=char_set)
        print(f"Loaded {data_subset.shape[0]} samples")
        
        # Create train/val/test splits
        train_data, val_data, test_data = create_data_splits(data_subset)
        
        # Display some samples
        random_indices = np.random.choice(len(train_data), size=10, replace=False)
        display_binary_images(train_data[random_indices], n_cols=5, figsize=(10, 5),
                            titles=[f"Char {char_name} - {i}" for i in range(10)],
                            save_path=f"results/plots/dbn_orig_samples_chars_{char_name}.png")
        
        # Adjust layer sizes if needed
        if layer_sizes is None:
            current_layer_sizes = [train_data.shape[1], 100, 100]
        else:
            current_layer_sizes = [train_data.shape[1]] + layer_sizes[1:]
        
        # Train DBN with current character set
        dbn_chars = train_dbn(
            data=train_data,
            val_data=val_data,
            layer_sizes=current_layer_sizes,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            k=k,
            model_name=f"dbn_alpha_chars_{char_name}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store pretraining losses if available
        if hasattr(dbn_chars, 'pretrain_losses') and dbn_chars.pretrain_losses:
            losses_by_chars[char_name] = dbn_chars.pretrain_losses

    return losses_by_chars

def experiment_learning_rate(train_data, val_data=None, learning_rates=[0.001, 0.01, 0.05, 0.1], 
                           layer_sizes=None, nb_epochs=100, batch_size=10, k=1, chars=None):
    """
    Experiment with varying learning rates in DBN.
    
    Parameters:
    -----------
    train_data: ndarray
        Training data
    val_data: ndarray, optional
        Validation data
    learning_rates: list
        List of learning rates to try
    layer_sizes, nb_epochs, batch_size, k: 
        Training parameters
    chars: list
        List of characters used (for logging)
        
    Returns:
    --------
    losses_by_lr: dict
        Dictionary mapping learning rates to pretraining losses
    best_lr: float
        Learning rate with lowest pretraining loss
    """
    print("\n========= Experiment: Varying the learning rate =========")
    losses_by_lr = {}
    best_loss = float('inf')
    best_lr = None

    print(f"Running experiment with learning rates: {learning_rates}")
    if chars:
        print(f"Using characters: {chars}")

    if layer_sizes is None:
        layer_sizes = [train_data.shape[1], 100, 100]

    for lr in learning_rates:
        print(f"\nTraining DBN with learning rate: {lr}")
        
        # Train DBN with current learning rate
        dbn_lr = train_dbn(
            data=train_data,
            val_data=val_data,
            layer_sizes=layer_sizes,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            k=k,
            model_name=f"dbn_alpha_lr{lr}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store pretraining losses if available
        if hasattr(dbn_lr, 'pretrain_losses') and dbn_lr.pretrain_losses:
            final_loss = dbn_lr.pretrain_losses[-1][-1]
            losses_by_lr[lr] = dbn_lr.pretrain_losses
            
            # Check if this is the best model based on final loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr

    print(f"Best learning rate: {best_lr} with pretraining loss: {best_loss:.6f}")
    return losses_by_lr, best_lr

def experiment_batch_size(train_data, val_data=None, batch_sizes=[5, 10, 20, 50], 
                        layer_sizes=None, nb_epochs=100, learning_rate=0.1, k=1, chars=None):
    """
    Experiment with varying batch sizes in DBN.
    
    Parameters:
    -----------
    train_data: ndarray
        Training data
    val_data: ndarray, optional
        Validation data
    batch_sizes: list
        List of batch sizes to try
    layer_sizes, nb_epochs, learning_rate, k: 
        Training parameters
    chars: list
        List of characters used (for logging)
        
    Returns:
    --------
    losses_by_batch: dict
        Dictionary mapping batch sizes to pretraining losses
    best_bs: int
        Batch size with lowest pretraining loss
    """
    print("\n========= Experiment: Varying the batch size =========")
    losses_by_batch = {}
    best_loss = float('inf')
    best_bs = None

    print(f"Running experiment with batch sizes: {batch_sizes}")
    if chars:
        print(f"Using characters: {chars}")

    if layer_sizes is None:
        layer_sizes = [train_data.shape[1], 100, 100]

    for bs in batch_sizes:
        print(f"\nTraining DBN with batch size: {bs}")
        
        # Train DBN with current batch size
        dbn_bs = train_dbn(
            data=train_data,
            val_data=val_data,
            layer_sizes=layer_sizes,
            nb_epochs=nb_epochs,
            batch_size=bs,
            learning_rate=learning_rate,
            k=k,
            model_name=f"dbn_alpha_bs{bs}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store pretraining losses if available
        if hasattr(dbn_bs, 'pretrain_losses') and dbn_bs.pretrain_losses:
            final_loss = dbn_bs.pretrain_losses[-1][-1]
            losses_by_batch[bs] = dbn_bs.pretrain_losses
            
            # Check if this is the best model based on final loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_bs = bs

    print(f"Best batch size: {best_bs} with pretraining loss: {best_loss:.6f}")
    return losses_by_batch, best_bs

def main():
    # Default parameters
    nb_epochs = 100
    batch_size = 10
    learning_rate = 0.1
    k = 1
    chars = [2, 3, 4, 5, 6] # Characters to load

    # Load data
    print("Loading Binary AlphaDigits dataset...")
    data, labels = load_binary_alphadigits(chars=chars, return_labels=True) 

    if data is None:
        print("Failed to load data. Exiting.")
        exit()

    print(f"Loaded {data.shape[0]} samples with dimension {data.shape[1]}")

    # Create train/validation/test splits
    train_data, val_data, test_data, train_labels, val_labels, test_labels = create_data_splits(data, labels)
    
    # Display some random original samples
    print("Displaying original samples:")
    random_indices = np.random.choice(len(train_data), size=10, replace=False)
    random_samples = train_data[random_indices]
    random_labels = train_labels[random_indices]
    display_binary_images(random_samples, n_cols=5, figsize=(10, 5), 
                         titles=[f"Character: {label}" for label in random_labels])
    
    # Initial layer architecture
    layer_sizes = [data.shape[1], 100, 100]
    
    # Train initial DBN with default parameters
    print("\n========= Initial DBN Training with Default Parameters =========")
    dbn_initial = train_dbn(
        data=train_data,
        val_data=val_data,
        layer_sizes=layer_sizes,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=k,
        model_name="dbn_alpha_initial",
        verbose=True,
    )
    
    # Perform experiments to find optimal parameters
    print("\n========= Running Experiments to Find Optimal Parameters =========")
    
    # Experiment with hidden dimensions
    hidden_dims = [50, 100, 200, 400]
    _, best_hidden = experiment_hidden_dimensions(
        train_data=train_data,
        val_data=val_data,
        hidden_dims=hidden_dims,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=k,
        chars=chars
    )

    # Experiment with number of layers
    num_layers = [1, 2, 3, 4]
    _, best_num_layers = experiment_number_of_layers(
        train_data=train_data,
        val_data=val_data,
        num_layers=num_layers,
        hidden_size=100,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=k,
        chars=chars
    )

    # Experiment with learning rates
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    _, best_lr = experiment_learning_rate(
        train_data=train_data,
        val_data=val_data,
        learning_rates=learning_rates,
        layer_sizes=layer_sizes,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        k=k,
        chars=chars
    )
    
    # Experiment with batch sizes
    batch_sizes = [10, 20, 50]
    _, best_bs = experiment_batch_size(
        train_data=train_data,
        val_data=val_data,
        batch_sizes=batch_sizes,
        layer_sizes=layer_sizes,
        nb_epochs=nb_epochs,
        learning_rate=learning_rate,
        k=k,
        chars=chars
    )
    
    # Experiment with different character sets
    char_sets = {
        "2-3": [2, 3],      # Two characters
        "2-4": [2, 3, 4],  # Three characters
        "2-6": [2, 3, 4, 5, 6],  # Five characters
        "0-9": list(range(10)),  # Ten characters
    }
    experiment_characters(
        char_sets=char_sets,
        layer_sizes=layer_sizes,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=k
    )
    
    # Train final model with best parameters
    print("\n========= Final DBN Training with Best Parameters =========")
    print(f"Using best parameters: hidden units={best_hidden}, num layers={best_num_layers}, learning rate={best_lr}, batch size={best_bs}")
    
    # Combine train and validation for final training
    train_final = np.concatenate([train_data, val_data], axis=0)
    
    # Create final layer architecture with best parameters
    final_layer_sizes = [train_final.shape[1]] + [best_hidden] * best_num_layers
    
    dbn_final = train_dbn(
        data=train_final,
        val_data=test_data,
        layer_sizes=final_layer_sizes,
        nb_epochs=nb_epochs,
        batch_size=best_bs,
        learning_rate=best_lr,
        k=k,
        model_name="dbn_alpha_final",
        verbose=True
    )
    
    # Evaluate on test data
    test_samples = dbn_final.generate_samples(n_samples=25, gibbs_steps=200)
    display_binary_images(test_samples, n_cols=5, figsize=(10, 5),
                        titles=[f"Final Model Sample {i}" for i in range(10)],
                        save_path="results/plots/dbn_final_test_samples.png")
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()