import numpy as np
import os
import pickle
from models import RBM
from utils import (load_binary_alphadigits,
                   display_binary_images,
                   display_weights,
                   plot_losses,
                   plot_comparison_losses)
from sklearn.model_selection import train_test_split

# Create directories for saving results if they don't exist
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

def create_data_splits(data, labels=None, val_size=0.1, test_size=0.2, random_state=42):
    """
    Create train/validation/test splits for the data.
    
    Parameters:
    -----------
    data: ndarray
        Input data to split
    labels: ndarray, optional
        Labels corresponding to data samples
    val_size: float
        Proportion of data to use for validation
    test_size: float
        Proportion of data to use for testing
    random_state: int
        Random seed for reproducibility
        
    Returns:
    --------
    train_data: ndarray
        Training data
    val_data: ndarray
        Validation data
    test_data: ndarray
        Test data
    train_labels, val_labels, test_labels: ndarray, optional
        Corresponding labels for each split if labels were provided
    """
    # First split data into temp (train+val) and test
    if labels is not None:
        temp_data, test_data, temp_labels, test_labels = train_test_split(
            data, labels, test_size=test_size, random_state=random_state)
        
        # Then split temp data into train and validation
        # Adjust val_size to account for the already removed test data
        adjusted_val_size = val_size / (1 - test_size)
        train_data, val_data, train_labels, val_labels = train_test_split(
            temp_data, temp_labels, test_size=adjusted_val_size, random_state=random_state)
        
        print(f"Data splits: Train: {train_data.shape[0]}, Validation: {val_data.shape[0]}, Test: {test_data.shape[0]}")
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    else:
        # Original behavior when no labels are provided
        temp_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        
        adjusted_val_size = val_size / (1 - test_size)
        train_data, val_data = train_test_split(temp_data, test_size=adjusted_val_size, random_state=random_state)
        
        print(f"Data splits: Train: {train_data.shape[0]}, Validation: {val_data.shape[0]}, Test: {test_data.shape[0]}")
        return train_data, val_data, test_data

def train_rbm(data, val_data=None, n_hidden=100, nb_epochs=100, batch_size=10, 
            learning_rate=0.1, k=1, model_name="rbm", verbose=True, save_model=True,
            save_samples=True, save_weights=True, plot=True):
    """
    Train an RBM model and optionally visualize results.
    
    Parameters:
    -----------
    data: ndarray
        Training data of shape (n_samples, n_features)
    val_data: ndarray, optional
        Validation data for monitoring reconstruction loss on unseen data
    n_hidden: int
        Number of hidden units
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
    rbm: RBM
        Trained RBM model
    """
    # Initialize and train RBM
    if verbose:
        print(f"Initializing RBM with {data.shape[1]} visible and {n_hidden} hidden units")
    
    rbm = RBM(n_visible=data.shape[1], n_hidden=n_hidden)

    # Train the model
    if verbose:
        print(f"Training RBM for {nb_epochs} epochs...")
    
    rbm.fit(data,
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
            print(f"Saving trained RBM model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(rbm, f)

    # Generate and display samples
    if save_samples:
        if verbose:
            print("Generating samples from the trained RBM...")
        samples = rbm.generate_samples(n_samples=25, gibbs_steps=200)
        
        if verbose:
            print("Displaying generated samples:")
        display_binary_images(samples, n_cols=5, figsize=(10, 5),
                             save_path=f"results/plots/{model_name}_samples.png")
    if plot:
        # Plot reconstruction loss during training
        if verbose:
            print("Plotting reconstruction loss:")
        
        # If we have validation losses, plot both training and validation
        if rbm.val_losses:
            losses = [rbm.losses, rbm.val_losses]
            layer_names = ["Training", "Validation"]
            plot_losses(losses, 
                    title='RBM Reconstruction Loss',
                    ylabel='Mean Squared Loss',
                    layer_names=layer_names,
                    save_path=f"results/plots/{model_name}_loss.png")
        else:
            # Otherwise just plot training loss
            plot_losses(rbm.losses, 
                    title='RBM Reconstruction Loss',
                    ylabel='Mean Squared Loss',
                    save_path=f"results/plots/{model_name}_loss.png")

    # Plot weights
    if save_weights:
        if verbose:
            print("Displaying RBM weights:")
        display_weights(rbm, height=20, width=16, figsize=(10, 10), n_cols=10,
                        save_path=f"results/plots/{model_name}_weights.png")
    
    return rbm

def experiment_hidden_dimensions(train_data, val_data=None, hidden_dims=[50, 100, 200, 400], 
                               nb_epochs=100, batch_size=10, learning_rate=0.1, k=1, chars=None):
    """
    Experiment with varying hidden dimensions in RBM.
    
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
        Hidden dimension with lowest validation loss (or training loss if no validation)
    """
    print(f"\n========= Experiment: Varying the hidden dimension =========")
    losses_by_dim = {}
    val_losses_by_dim = {}
    best_loss = float('inf')
    best_dim = None

    print(f"Running experiment with hidden dimensions: {hidden_dims}")
    if chars:
        print(f"Using characters: {chars}")

    for dim in hidden_dims:
        print(f"\nTraining RBM with {dim} hidden units...")
        
        # Train RBM with current hidden dimension
        rbm_dim = train_rbm(
            data=train_data,
            val_data=val_data,
            n_hidden=dim,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            k=k,
            model_name=f"rbm_alpha_h{dim}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store reconstruction losses
        losses_by_dim[dim] = rbm_dim.losses
        if rbm_dim.val_losses:
            val_losses_by_dim[dim] = rbm_dim.val_losses
            # Check if this is the best model based on final validation loss
            if rbm_dim.val_losses[-1] < best_loss:
                best_loss = rbm_dim.val_losses[-1]
                best_dim = dim
        else:
            # If no validation data, use training loss
            if rbm_dim.losses[-1] < best_loss:
                best_loss = rbm_dim.losses[-1]
                best_dim = dim

    # Plot comparative reconstruction losses for training
    plot_comparison_losses(
        losses_dict=losses_by_dim,
        xlabel='Epoch',
        ylabel='Reconstruction Loss',
        title='RBM Training Reconstruction Loss for Different Hidden Dimensions',
        label_prefix='Hidden Units: ',
        save_path="results/plots/rbm_hidden_dim_train_comparison.png"
    )
    
    # Plot comparative validation losses if available
    if val_losses_by_dim:
        plot_comparison_losses(
            losses_dict=val_losses_by_dim,
            xlabel='Epoch',
            ylabel='Validation Reconstruction Loss',
            title='RBM Validation Reconstruction Loss for Different Hidden Dimensions',
            label_prefix='Hidden Units: ',
            save_path="results/plots/rbm_hidden_dim_val_comparison.png"
        )
    
    print(f"Best hidden dimension: {best_dim} with {'validation' if rbm_dim.val_losses else 'training'} loss: {best_loss:.6f}")
    return losses_by_dim, best_dim

def experiment_characters(char_sets, n_hidden=100, nb_epochs=100, batch_size=10, 
                        learning_rate=0.1, k=1):
    """
    Experiment with varying character sets in RBM.
    
    Parameters:
    -----------
    char_sets: dict
        Dictionary mapping character set names to lists of characters
    n_hidden, nb_epochs, batch_size, learning_rate, k: 
        Training parameters
        
    Returns:
    --------
    losses_by_chars: dict
        Dictionary mapping character sets to reconstruction losses
    """
    print("\n========= Experiment: Varying the number of characters =========")
    losses_by_chars = {}
    val_losses_by_chars = {}

    for char_name, char_set in char_sets.items():
        print(f"\nTraining RBM with characters: {char_name}")
        
        # Load data for current character set
        data_subset = load_binary_alphadigits(chars=char_set)
        print(f"Loaded {data_subset.shape[0]} samples")
        
        # Create train/val/test splits
        train_data, val_data, test_data = create_data_splits(data_subset)
        
        # Display some samples
        random_indices = np.random.choice(len(train_data), size=10, replace=False)
        display_binary_images(train_data[random_indices], n_cols=5, figsize=(10, 5),
                            titles=[f"Char {char_name} - {i}" for i in range(10)],
                            save_path=f"results/plots/orig_samples_chars_{char_name}.png")
        
        # Train RBM with current character set
        rbm_chars = train_rbm(
            data=train_data,
            val_data=val_data,
            n_hidden=n_hidden,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            k=k,
            model_name=f"rbm_alpha_chars_{char_name}",
            verbose=True,
            save_weights=False,
            plot=False,
        )
        
        # Store reconstruction losses
        losses_by_chars[char_name] = rbm_chars.losses
        if rbm_chars.val_losses:
            val_losses_by_chars[char_name] = rbm_chars.val_losses

    # Plot comparative reconstruction losses for training
    plot_comparison_losses(
        losses_dict=losses_by_chars,
        xlabel='Epoch',
        ylabel='Training Reconstruction Loss',
        title='RBM Training Reconstruction Loss for Different Character Sets',
        label_prefix='Characters: ',
        save_path="results/plots/rbm_char_sets_train_comparison.png"
    )
    
    # Plot comparative validation losses if available
    if val_losses_by_chars:
        plot_comparison_losses(
            losses_dict=val_losses_by_chars,
            xlabel='Epoch',
            ylabel='Validation Reconstruction Loss',
            title='RBM Validation Reconstruction Loss for Different Character Sets',
            label_prefix='Characters: ',
            save_path="results/plots/rbm_char_sets_val_comparison.png"
        )
    
    return losses_by_chars

def experiment_learning_rate(train_data, val_data=None, learning_rates=[0.001, 0.01, 0.05, 0.1], 
                           n_hidden=100, nb_epochs=100, batch_size=10, k=1, chars=None):
    """
    Experiment with varying learning rates in RBM.
    
    Parameters:
    -----------
    train_data: ndarray
        Training data
    val_data: ndarray, optional
        Validation data
    learning_rates: list
        List of learning rates to try
    n_hidden, nb_epochs, batch_size, k: 
        Training parameters
    chars: list
        List of characters used (for logging)
        
    Returns:
    --------
    losses_by_lr: dict
        Dictionary mapping learning rates to reconstruction losses
    best_lr: float
        Learning rate with lowest validation loss (or training loss if no validation)
    """
    print("\n========= Experiment: Varying the learning rate =========")
    losses_by_lr = {}
    val_losses_by_lr = {}
    best_loss = float('inf')
    best_lr = None

    print(f"Running experiment with learning rates: {learning_rates}")
    if chars:
        print(f"Using characters: {chars}")

    for lr in learning_rates:
        print(f"\nTraining RBM with learning rate: {lr}")
        
        # Train RBM with current learning rate
        rbm_lr = train_rbm(
            data=train_data,
            val_data=val_data,
            n_hidden=n_hidden,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            k=k,
            model_name=f"rbm_alpha_lr{lr}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store reconstruction losses
        losses_by_lr[lr] = rbm_lr.losses
        if rbm_lr.val_losses:
            val_losses_by_lr[lr] = rbm_lr.val_losses
            # Check if this is the best model based on final validation loss
            if rbm_lr.val_losses[-1] < best_loss:
                best_loss = rbm_lr.val_losses[-1]
                best_lr = lr
        else:
            # If no validation data, use training loss
            if rbm_lr.losses[-1] < best_loss:
                best_loss = rbm_lr.losses[-1]
                best_lr = lr

    # Plot comparative reconstruction losses for training
    plot_comparison_losses(
        losses_dict=losses_by_lr,
        xlabel='Epoch',
        ylabel='Training Reconstruction Loss',
        title='RBM Training Reconstruction Loss for Different Learning Rates',
        label_prefix='Learning Rate: ',
        save_path="results/plots/rbm_learning_rate_train_comparison.png"
    )
    
    # Plot comparative validation losses if available
    if val_losses_by_lr:
        plot_comparison_losses(
            losses_dict=val_losses_by_lr,
            xlabel='Epoch',
            ylabel='Validation Reconstruction Loss',
            title='RBM Validation Reconstruction Loss for Different Learning Rates',
            label_prefix='Learning Rate: ',
            save_path="results/plots/rbm_learning_rate_val_comparison.png"
        )
    
    print(f"Best learning rate: {best_lr} with {'validation' if rbm_lr.val_losses else 'training'} loss: {best_loss:.6f}")
    return losses_by_lr, best_lr

def experiment_batch_size(train_data, val_data=None, batch_sizes=[5, 10, 20, 50], 
                        n_hidden=100, nb_epochs=100, learning_rate=0.1, k=1, chars=None):
    """
    Experiment with varying batch sizes in RBM.
    
    Parameters:
    -----------
    train_data: ndarray
        Training data
    val_data: ndarray, optional
        Validation data
    batch_sizes: list
        List of batch sizes to try
    n_hidden, nb_epochs, learning_rate, k: 
        Training parameters
    chars: list
        List of characters used (for logging)
        
    Returns:
    --------
    losses_by_batch: dict
        Dictionary mapping batch sizes to reconstruction losses
    best_bs: int
        Batch size with lowest validation loss (or training loss if no validation)
    """
    print("\n========= Experiment: Varying the batch size =========")
    losses_by_batch = {}
    val_losses_by_batch = {}
    best_loss = float('inf')
    best_bs = None

    print(f"Running experiment with batch sizes: {batch_sizes}")
    if chars:
        print(f"Using characters: {chars}")

    for bs in batch_sizes:
        print(f"\nTraining RBM with batch size: {bs}")
        
        # Train RBM with current batch size
        rbm_bs = train_rbm(
            data=train_data,
            val_data=val_data,
            n_hidden=n_hidden,
            nb_epochs=nb_epochs,
            batch_size=bs,
            learning_rate=learning_rate,
            k=k,
            model_name=f"rbm_alpha_bs{bs}",
            verbose=True,
            save_weights=False,
            plot=False
        )
        
        # Store reconstruction losses
        losses_by_batch[bs] = rbm_bs.losses
        if rbm_bs.val_losses:
            val_losses_by_batch[bs] = rbm_bs.val_losses
            # Check if this is the best model based on final validation loss
            if rbm_bs.val_losses[-1] < best_loss:
                best_loss = rbm_bs.val_losses[-1]
                best_bs = bs
        else:
            # If no validation data, use training loss
            if rbm_bs.losses[-1] < best_loss:
                best_loss = rbm_bs.losses[-1]
                best_bs = bs

    # Plot comparative reconstruction losses for training
    plot_comparison_losses(
        losses_dict=losses_by_batch,
        xlabel='Epoch',
        ylabel='Training Reconstruction Loss',
        title='RBM Training Reconstruction Loss for Different Batch Sizes',
        label_prefix='Batch Size: ',
        save_path="results/plots/rbm_batch_size_train_comparison.png"
    )
    
    # Plot comparative validation losses if available
    if val_losses_by_batch:
        plot_comparison_losses(
            losses_dict=val_losses_by_batch,
            xlabel='Epoch',
            ylabel='Validation Reconstruction Loss',
            title='RBM Validation Reconstruction Loss for Different Batch Sizes',
            label_prefix='Batch Size: ',
            save_path="results/plots/rbm_batch_size_val_comparison.png"
        )
    
    print(f"Best batch size: {best_bs} with {'validation' if rbm_bs.val_losses else 'training'} loss: {best_loss:.6f}")
    return losses_by_batch, best_bs

def main():
    # Default parameters
    n_hidden = 100
    nb_epochs = 100
    batch_size = 10
    learning_rate = 0.1
    k = 1
    chars = [2, 3, 4] # Characters to load

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
    
    # Train initial RBM with default parameters
    print("\n========= Initial RBM Training with Default Parameters =========")
    rbm_initial = train_rbm(
        data=train_data,
        val_data=val_data,
        n_hidden=n_hidden,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=k,
        model_name="rbm_alpha_initial",
        verbose=True
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

    # Experiment with learning rates
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    _, best_lr = experiment_learning_rate(
        train_data=train_data,
        val_data=val_data,
        learning_rates=learning_rates,
        n_hidden=n_hidden,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        k=k,
        chars=chars
    )

    # Experiment with batch sizes
    batch_sizes = [5, 10, 20, 50]
    _, best_bs = experiment_batch_size(
        train_data=train_data,
        val_data=val_data,
        batch_sizes=batch_sizes,
        n_hidden=n_hidden,
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
        #"0-9;A-J": list(range(20)),  # Twenty characters
        #"all_chars": list(range(36))  # All thirty-six characters
    }
    experiment_characters(
        char_sets=char_sets,
        n_hidden=n_hidden,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=k
    )
    
    # Train final model with best parameters
    print("\n========= Final RBM Training with Best Parameters =========")
    print(f"Using best parameters: hidden units={best_hidden}, learning rate={best_lr}, batch size={best_bs}")
    
    # Combine train and validation for final training
    train_final = np.concatenate([train_data, val_data], axis=0)
    
    rbm_final = train_rbm(
        data=train_final,
        val_data=test_data,
        n_hidden=best_hidden,
        nb_epochs=nb_epochs,
        batch_size=best_bs,
        learning_rate=best_lr,
        k=k,
        model_name="rbm_alpha_final",
        verbose=True,
        save_samples=True,
        plot=True,
        save_weights=True
    )
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()