import numpy as np
import os
import pickle
from joblib import Parallel, delayed
from models import DNN, DBN
from utils import (load_mnist,
                   plot_comparison,
                   display_binary_images)

# Create directories for saving results if they don't exist
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def train_and_evaluate(X_train, y_train_onehot, X_test, y_test_onehot, 
                      layer_sizes, use_pretraining=False, 
                      pretrain_epochs=100, train_epochs=200, batch_size=100,
                      learning_rate=0.1, verbose=True, save_model=False, model_name="dnn"):
    """
    Train a DNN with or without pre-training and evaluate it.
    
    Parameters:
    -----------
    X_train: array-like
        Training data
    y_train_onehot: array-like
        One-hot encoded training labels
    X_test: array-like
        Test data
    y_test_onehot: array-like
        One-hot encoded test labels
    layer_sizes: list
        List of layer sizes [input_size, hidden1_size, ..., output_size]
    use_pretraining: bool
        Whether to use pre-training
    pretrain_epochs: int
        Number of epochs for pre-training
    train_epochs: int
        Number of epochs for supervised training
    batch_size: int
        Batch size
    learning_rate: float
        Learning rate
    verbose: bool
        Whether to print progress
    save_model: bool
        Whether to save the trained model
    model_name: str
        Base name to use for saving the model
    
    Returns:
    --------
    tuple
        (DNN, training_history, train_error, test_error)
    """
    if use_pretraining:
        if verbose:
            print(f"Pretraining DBN with {len(layer_sizes)-2} hidden layers...")
        
        # Initialize and pretrain DBN (without output layer)
        dbn = DBN(layer_sizes[:-1])
        dbn.fit(X_train,
                nb_epochs=pretrain_epochs,
                batch_size=batch_size,
                lr=learning_rate,
                verbose=verbose)
        
        if verbose:
            print(f"Initializing DNN with pre-trained weights...")
        # Initialize DNN with pretrained weights
        dnn = DNN(layer_sizes, dbn=dbn)
    else:
        if verbose:
            print(f"Initializing DNN with random weights...")
        # Initialize DNN with random weights
        dnn = DNN(layer_sizes)
    
    # Train the DNN
    if verbose:
        print(f"Training DNN for {train_epochs} epochs...")
    errors = dnn.fit(X_train,
                      y_train_onehot,
                      nb_epochs=train_epochs,
                      batch_size=batch_size,
                      lr=learning_rate,
                      verbose=verbose)
    
    # Evaluate the DNN
    train_error = dnn.error_rate(X_train, y_train_onehot)
    test_error = dnn.error_rate(X_test, y_test_onehot)
    
    if verbose:
        print(f"Training error: {train_error:.4f}")
        print(f"Test error: {test_error:.4f}")
    
    # Save model if requested
    if save_model:
        model_path = f"results/models/{model_name}_{'pretrained' if use_pretraining else 'random'}.pkl"
        print(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(dnn, f)
        
    return dnn, errors, train_error, test_error

def compare_layer_count(X_train, y_train_onehot, X_test, y_test_onehot, 
                       base_neurons=200, max_layers=5):
    """
    Compare network performance with different number of layers.
    
    Parameters:
    -----------
    X_train, y_train_onehot, X_test, y_test_onehot:
        Training and testing data
    base_neurons: int
        Number of neurons per hidden layer
    max_layers: int
        Maximum number of hidden layers to test
    
    Returns:
    --------
    tuple
        (layer_counts, pretrained_train_errors, pretrained_test_errors, random_train_errors, random_test_errors)
    """
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    layer_counts = list(range(2, max_layers + 1))  # Number of hidden layers
    
    # Define a function to train one configuration in parallel
    def train_for_layer_count(n_layers):
        print(f"\n=== Testing with {n_layers} hidden layers ===")
        
        # Create layer sizes [input, hidden1, hidden2, ..., output]
        layer_sizes = [input_size] + [base_neurons] * n_layers + [output_size]
        
        print("Training with pre-training...")
        _, _, pretrained_train_error, pretrained_test_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=True, 
            pretrain_epochs=100, train_epochs=200, verbose=True,
            save_model=True, model_name=f"dnn_layers_{n_layers}_pretrained")
        
        print("Training with random initialization...")
        _, _, random_train_error, random_test_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=False,
            train_epochs=200, verbose=True,
            save_model=True, model_name=f"dnn_layers_{n_layers}_random")
        
        print(f"Pre-trained train error: {pretrained_train_error:.4f}, test error: {pretrained_test_error:.4f}")
        print(f"Random init train error: {random_train_error:.4f}, test error: {random_test_error:.4f}")
        
        return (pretrained_train_error, pretrained_test_error, random_train_error, random_test_error)
    
    # Execute the tasks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_for_layer_count)(n_layers) for n_layers in layer_counts
    )
    
    # Extract results
    pretrained_train_errors = [res[0] for res in results]
    pretrained_test_errors = [res[1] for res in results]
    random_train_errors = [res[2] for res in results]
    random_test_errors = [res[3] for res in results]
    
    # Use the plot function - assuming it accepts multiple lines
    plot_comparison(
        layer_counts, 
        pretrained_test_errors,
        random_test_errors,
        pretrained_train_errors,
        random_train_errors,
        "Number of Hidden Layers",
        "Error Rate",
        "Effect of Network Depth on Error Rate",
        legend_labels=["Pretrained (Test)", "Random Init (Test)", 
                      "Pretrained (Train)", "Random Init (Train)"],
        save_path="results/plots/layer_count_comparison"
    )
    
    return layer_counts, pretrained_train_errors, pretrained_test_errors, random_train_errors, random_test_errors

def compare_neuron_count(X_train, y_train_onehot, X_test, y_test_onehot, 
                        n_layers=2, neuron_counts=None):
    """
    Compare network performance with different number of neurons per layer.
    
    Parameters:
    -----------
    X_train, y_train_onehot, X_test, y_test_onehot:
        Training and testing data
    n_layers: int
        Number of hidden layers
    neuron_counts: list or None
        List of neuron counts to test
    
    Returns:
    --------
    tuple
        (neuron_counts, pretrained_train_errors, pretrained_test_errors, random_train_errors, random_test_errors)
    """
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    if neuron_counts is None:
        neuron_counts = [100, 200, 300, 400, 500, 700]
    
    # Define a function to train one configuration in parallel
    def train_for_neuron_count(n_neurons):
        print(f"\n=== Testing with {n_neurons} neurons per layer ===")
        
        # Create layer sizes [input, hidden1, hidden2, ..., output]
        layer_sizes = [input_size] + [n_neurons] * n_layers + [output_size]
        
        print("Training with pre-training...")
        _, _, pretrained_train_error, pretrained_test_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=True, 
            pretrain_epochs=100, train_epochs=200, verbose=True,
            save_model=True, model_name=f"dnn_neurons_{n_neurons}_pretrained")
        
        print("Training with random initialization...")
        _, _, random_train_error, random_test_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=False,
            train_epochs=200, verbose=True,
            save_model=True, model_name=f"dnn_neurons_{n_neurons}_random")
        
        print(f"Pre-trained train error: {pretrained_train_error:.4f}, test error: {pretrained_test_error:.4f}")
        print(f"Random init train error: {random_train_error:.4f}, test error: {random_test_error:.4f}")
        
        return (pretrained_train_error, pretrained_test_error, random_train_error, random_test_error)
    
    # Execute the tasks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_for_neuron_count)(n_neurons) for n_neurons in neuron_counts
    )
    
    # Extract results
    pretrained_train_errors = [res[0] for res in results]
    pretrained_test_errors = [res[1] for res in results]
    random_train_errors = [res[2] for res in results]
    random_test_errors = [res[3] for res in results]
    
    # Use the plot function
    plot_comparison(
        neuron_counts, 
        pretrained_test_errors,
        random_test_errors,
        pretrained_train_errors,
        random_train_errors,
        "Number of Neurons per Layer", 
        "Error Rate",
        "Effect of Layer Width on Error Rate",
        legend_labels=["Pretrained (Test)", "Random Init (Test)", 
                      "Pretrained (Train)", "Random Init (Train)"],
        save_path="results/plots/neuron_count_comparison"
    )
    
    return neuron_counts, pretrained_train_errors, pretrained_test_errors, random_train_errors, random_test_errors

def compare_training_size(X_train, y_train_onehot, X_test, y_test_onehot,
                         sample_sizes=None, n_layers=2, n_neurons=200):
    """
    Compare network performance with different training set sizes.
    
    Parameters:
    -----------
    X_train, y_train_onehot, X_test, y_test_onehot:
        Training and testing data
    sample_sizes: list or None
        List of sample sizes to test
    n_layers: int
        Number of hidden layers
    n_neurons: int
        Number of neurons per hidden layer
    
    Returns:
    --------
    tuple
        (sample_sizes, pretrained_train_errors, pretrained_test_errors, random_train_errors, random_test_errors)
    """
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    if sample_sizes is None:
        sample_sizes = [1000, 3000, 7000, 10000, 30000, 60000]
    
    # Define a function to train one configuration in parallel
    def train_for_sample_size(size):
        if size > X_train.shape[0]:
            size = X_train.shape[0]
            
        print(f"\n=== Testing with {size} training samples ===")
        
        # Subsample the data
        indices = np.random.choice(X_train.shape[0], size, replace=False)
        X_subset = X_train[indices]
        y_subset_onehot = y_train_onehot[indices]
        
        # Create layer sizes [input, hidden1, hidden2, ..., output]
        layer_sizes = [input_size] + [n_neurons] * n_layers + [output_size]
        
        print("Training with pre-training...")
        _, _, pretrained_train_error, pretrained_test_error = train_and_evaluate(
            X_subset, y_subset_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=True, 
            pretrain_epochs=100, train_epochs=200, verbose=True,
            save_model=True, model_name=f"dnn_trainsize_{size}_pretrained")
        
        print("Training with random initialization...")
        _, _, random_train_error, random_test_error = train_and_evaluate(
            X_subset, y_subset_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=False,
            train_epochs=200, verbose=True,
            save_model=True, model_name=f"dnn_trainsize_{size}_random")
        
        print(f"Pre-trained train error: {pretrained_train_error:.4f}, test error: {pretrained_test_error:.4f}")
        print(f"Random init train error: {random_train_error:.4f}, test error: {random_test_error:.4f}")
        
        return (pretrained_train_error, pretrained_test_error, random_train_error, random_test_error)
    
    # Execute the tasks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_for_sample_size)(size) for size in sample_sizes
    )
    
    # Extract results
    pretrained_train_errors = [res[0] for res in results]
    pretrained_test_errors = [res[1] for res in results]
    random_train_errors = [res[2] for res in results]
    random_test_errors = [res[3] for res in results]
    
    # Use the plot function
    plot_comparison(
        sample_sizes, 
        pretrained_test_errors,
        random_test_errors,
        pretrained_train_errors,
        random_train_errors,
        "Training Set Size", 
        "Error Rate",
        "Effect of Training Set Size on Error Rate",
        legend_labels=["Pretrained (Test)", "Random Init (Test)", 
                      "Pretrained (Train)", "Random Init (Train)"],
        save_path="results/plots/training_size_comparison"
    )
    
    return sample_sizes, pretrained_train_errors, pretrained_test_errors, random_train_errors, random_test_errors

def run_hyperparameter_experiments(X_train, y_train_onehot, X_test, y_test_onehot):
    """Run hyperparameter experiments and return the results."""
    print("\nRunning hyperparameter experiments...")
    
    # Compare number of layers
    print("\n=== EXPERIMENT 1: COMPARING NUMBER OF LAYERS ===")
    layer_counts, pretrained_layer_train_errors, pretrained_layer_test_errors, random_layer_train_errors, random_layer_test_errors = compare_layer_count(
        X_train, y_train_onehot, X_test, y_test_onehot)
    
    # Compare number of neurons per layer
    print("\n=== EXPERIMENT 2: COMPARING NUMBER OF NEURONS PER LAYER ===")
    neuron_counts, pretrained_neuron_train_errors, pretrained_neuron_test_errors, random_neuron_train_errors, random_neuron_test_errors = compare_neuron_count(
        X_train, y_train_onehot, X_test, y_test_onehot)
    
    # Compare training set size
    print("\n=== EXPERIMENT 3: COMPARING TRAINING SET SIZE ===")
    sample_sizes, pretrained_size_train_errors, pretrained_size_test_errors, random_size_train_errors, random_size_test_errors = compare_training_size(
        X_train, y_train_onehot, X_test, y_test_onehot)
    
    print("\nExperiments completed.")
    
    # Return all experiment results
    return {
        'layer_experiment': (layer_counts, pretrained_layer_train_errors, pretrained_layer_test_errors, random_layer_train_errors, random_layer_test_errors),
        'neuron_experiment': (neuron_counts, pretrained_neuron_train_errors, pretrained_neuron_test_errors, random_neuron_train_errors, random_neuron_test_errors),
        'size_experiment': (sample_sizes, pretrained_size_train_errors, pretrained_size_test_errors, random_size_train_errors, random_size_test_errors)
    }

def train_optimal_model(X_train, y_train_onehot,
                        X_test, y_test_onehot,
                        load_pretrained_dbn=False):
    """
    Train a model with optimal configuration and visualize results.
    
    Parameters:
    -----------
    X_train, y_train_onehot, X_test, y_test_onehot:
        Training and testing data
    load_pretrained_dbn:
        Whether to load a pre-trained DBN instead of training a new one
    
    Returns:
    --------
    DNN
        Trained DNN model
    """
    print("\nTraining DNN with optimal configuration and pre-training...")
    # Define the architecture
    layer_sizes = [784, 500, 500, 10]
    
    # Get pre-trained DBN (either by loading or training)
    if load_pretrained_dbn:
        dbn_path = f"results/models/optimal_dbn.pkl"
        try:
            print(f"Loading pre-trained DBN from {dbn_path}...")
            with open(dbn_path, 'rb') as f:
                dbn = pickle.load(f)
            print("Pre-trained DBN loaded successfully.")
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading pre-trained DBN: {e}")
            print("Falling back to training a new DBN.")
            # Pre-training with DBN
            print(f"Pretraining DBN with {len(layer_sizes)-2} hidden layers...")
            dbn = DBN(layer_sizes[:-1])
            dbn.fit(X_train,
                    nb_epochs=100,
                    batch_size=100,
                    lr=0.1,
                    verbose=True)
            # Save the pre-trained DBN
            print(f"Saving pre-trained DBN to {dbn_path}")
            with open(dbn_path, 'wb') as f:
                pickle.dump(dbn, f)
    else:
        # Pre-training with DBN
        print(f"Pretraining DBN with {len(layer_sizes)-2} hidden layers...")
        dbn = DBN(layer_sizes[:-1])
        dbn.fit(X_train,
                nb_epochs=100,
                batch_size=100,
                lr=0.1,
                verbose=True)
        # Save the pre-trained DBN
        dbn_path = f"results/models/optimal_dbn.pkl"
        print(f"Saving pre-trained DBN to {dbn_path}")
        with open(dbn_path, 'wb') as f:
            pickle.dump(dbn, f)
    
    # Initialize DNN with pre-trained weights
    print(f"Initializing DNN with pre-trained weights...")
    dnn = DNN(layer_sizes, dbn=dbn)
    
    # Fine-tune DNN with supervision
    print(f"Fine-tuning DNN...")
    
    # Split training data to create a validation set
    n_val = int(0.1 * X_train.shape[0])
    X_val_split = X_train[-n_val:]
    y_val_split = y_train_onehot[-n_val:]
    X_train_split = X_train[:-n_val]
    y_train_split = y_train_onehot[:-n_val]
    
    # Train the DNN
    dnn.fit(X_train_split, y_train_split,
            X_val_split, y_val_split,
            nb_epochs=100,
            batch_size=100,
            lr=0.1,
            verbose=True)
    
    # Evaluate the DNN
    train_error = dnn.error_rate(X_train, y_train_onehot)
    test_error = dnn.error_rate(X_test, y_test_onehot)
    
    print(f"Training error: {train_error:.4f}")
    print(f"Test error: {test_error:.4f}")
    
    # Save the model
    model_path = f"results/models/optimal_dnn.pkl"
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(dnn, f)
    
    print("\nOptimal model training completed.")
    
    return dnn

if __name__ == "__main__":
    # Load MNIST dataset (now always one-hot encoded)
    X_train, y_train_onehot, X_test, y_test_onehot = load_mnist(
        binarize_threshold=0.5, normalize=True, use_cache=True)
    
    # Run hyperparameter experiments
    run_hyperparameter_experiments(
        X_train, y_train_onehot, X_test, y_test_onehot)
    
    # Train the optimal model
    # Set load_pretrained_dbn to True to load a pre-trained DBN instead of training a new one
    #optimal_model = train_optimal_model(
    #    X_train, y_train_onehot, X_test, y_test_onehot,
    #    load_pretrained_dbn=False)  # Set to True to load a previously trained DBN