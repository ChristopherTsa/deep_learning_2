import numpy as np
import matplotlib.pyplot as plt
from models.dnn import DNN
from models.dbn import DBN
from utils.data_utils import load_mnist, one_hot_encode
from utils.visualization import plot_performance_comparison

# Set random seed for reproducibility
np.random.seed(42)

def train_and_evaluate(X_train, y_train_onehot, X_test, y_test_onehot, 
                      layer_sizes, use_pretraining=False, 
                      pretrain_epochs=100, train_epochs=200, batch_size=100,
                      learning_rate=0.1, verbose=True):
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
    
    Returns:
    --------
    tuple
        (DNN, training_history, test_error)
    """
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    if use_pretraining:
        if verbose:
            print(f"Pretraining DBN with {len(layer_sizes)-2} hidden layers...")
        
        # Initialize and pretrain DBN (without output layer)
        dbn = DBN(layer_sizes[:-1])
        dbn.fit(X_train, nb_epochs=pretrain_epochs, batch_size=batch_size,
               lr=learning_rate, verbose=verbose)
        
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
    history = dnn.fit(X_train, y_train_onehot, 
                     nb_epochs=train_epochs, batch_size=batch_size,
                     lr=learning_rate, verbose=verbose)
    
    # Evaluate the DNN
    train_error = dnn.error_rate(X_train, y_train_onehot)
    test_error = dnn.error_rate(X_test, y_test_onehot)
    
    if verbose:
        print(f"Training error: {train_error:.4f}")
        print(f"Test error: {test_error:.4f}")
        
    return dnn, history, test_error

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
        (layer_counts, pretrained_errors, random_errors)
    """
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    layer_counts = list(range(2, max_layers + 1))  # Number of hidden layers
    pretrained_errors = []
    random_errors = []
    
    for n_layers in layer_counts:
        print(f"\n=== Testing with {n_layers} hidden layers ===")
        
        # Create layer sizes [input, hidden1, hidden2, ..., output]
        layer_sizes = [input_size] + [base_neurons] * n_layers + [output_size]
        
        print("Training with pre-training...")
        _, _, pretrained_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=True, 
            pretrain_epochs=100, train_epochs=200, verbose=False)  # Ajusté aux valeurs recommandées
        
        print("Training with random initialization...")
        _, _, random_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=False,
            train_epochs=200, verbose=False)  # Ajusté à 200 époques
        
        pretrained_errors.append(pretrained_error)
        random_errors.append(random_error)
        
        print(f"Pre-trained error: {pretrained_error:.4f}")
        print(f"Random init error: {random_error:.4f}")
            
    return layer_counts, pretrained_errors, random_errors

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
        (neuron_counts, pretrained_errors, random_errors)
    """
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    if neuron_counts is None:
        neuron_counts = [100, 200, 300, 400, 500, 700]
        
    pretrained_errors = []
    random_errors = []
    
    for n_neurons in neuron_counts:
        print(f"\n=== Testing with {n_neurons} neurons per layer ===")
        
        # Create layer sizes [input, hidden1, hidden2, ..., output]
        layer_sizes = [input_size] + [n_neurons] * n_layers + [output_size]
        
        print("Training with pre-training...")
        _, _, pretrained_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=True, 
            pretrain_epochs=100, train_epochs=200, verbose=False)  # Ajusté aux valeurs recommandées
        
        print("Training with random initialization...")
        _, _, random_error = train_and_evaluate(
            X_train, y_train_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=False,
            train_epochs=200, verbose=False)  # Ajusté à 200 époques
        
        pretrained_errors.append(pretrained_error)
        random_errors.append(random_error)
        
        print(f"Pre-trained error: {pretrained_error:.4f}")
        print(f"Random init error: {random_error:.4f}")
            
    return neuron_counts, pretrained_errors, random_errors

def compare_training_size(X_train, y_train, y_train_onehot, X_test, y_test_onehot,
                         sample_sizes=None, n_layers=2, n_neurons=200):
    """
    Compare network performance with different training set sizes.
    
    Parameters:
    -----------
    X_train, y_train, y_train_onehot, X_test, y_test_onehot:
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
        (sample_sizes, pretrained_errors, random_errors)
    """
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    
    if sample_sizes is None:
        sample_sizes = [1000, 3000, 7000, 10000, 30000, 60000]
            
    pretrained_errors = []
    random_errors = []
    
    for size in sample_sizes:
        if size > X_train.shape[0]:
            size = X_train.shape[0]
            
        print(f"\n=== Testing with {size} training samples ===")
        
        # Subsample the data
        indices = np.random.choice(X_train.shape[0], size, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        y_subset_onehot = one_hot_encode(y_subset)
        
        # Create layer sizes [input, hidden1, hidden2, ..., output]
        layer_sizes = [input_size] + [n_neurons] * n_layers + [output_size]
        
        print("Training with pre-training...")
        _, _, pretrained_error = train_and_evaluate(
            X_subset, y_subset_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=True, 
            pretrain_epochs=100, train_epochs=200, verbose=False)  # Ajusté aux valeurs recommandées
        
        print("Training with random initialization...")
        _, _, random_error = train_and_evaluate(
            X_subset, y_subset_onehot, X_test, y_test_onehot, 
            layer_sizes, use_pretraining=False,
            train_epochs=200, verbose=False)  # Ajusté à 200 époques
        
        pretrained_errors.append(pretrained_error)
        random_errors.append(random_error)
        
        print(f"Pre-trained error: {pretrained_error:.4f}")
        print(f"Random init error: {random_error:.4f}")
            
    return sample_sizes, pretrained_errors, random_errors

def show_output_probabilities(dnn, X_test, y_test, num_samples=5):
    """
    Display output probabilities for a few test samples.
    
    Parameters:
    -----------
    dnn: DNN
        Trained neural network
    X_test: array-like
        Test data
    y_test: array-like
        Test labels (not one-hot encoded)
    num_samples: int
        Number of samples to display
    """
    # Select random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        x = X_test[idx].reshape(1, -1)
        true_label = y_test[idx]
        
        # Get network output probabilities
        probs = dnn.predict_proba(x)[0]
        
        # Print results
        print(f"\nSample {i+1} (True label: {true_label})")
        print("Class probabilities:")
        for j, p in enumerate(probs):
            marker = "*" if j == true_label else " "
            print(f"  Class {j}: {p:.4f} {marker}")
        
        # Optional: display the image
        plt.figure(figsize=(3, 3))
        plt.imshow(X_test[idx].reshape(28, 28), cmap='binary')
        plt.title(f"True: {true_label}, Pred: {np.argmax(probs)}")
        plt.axis('off')
        plt.show()

def run_experiments():
    """Run all experiments and generate plots."""
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist(binarize_threshold=0.5, normalize=True)
    
    # Convert pandas Series to numpy arrays if necessary
    if hasattr(y_train, 'to_numpy'):
        y_train = y_train.to_numpy()
    if hasattr(y_test, 'to_numpy'):
        y_test = y_test.to_numpy()
        
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)
    
    print(f"MNIST loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Compare number of layers
    print("\n=== EXPERIMENT 1: COMPARING NUMBER OF LAYERS ===")
    layer_counts, pretrained_layer_errors, random_layer_errors = compare_layer_count(
        X_train, y_train_onehot, X_test, y_test_onehot)
    
    plot_performance_comparison(
        layer_counts, pretrained_layer_errors, random_layer_errors,
        "Number of Hidden Layers", "Error Rate",
        "Effect of Network Depth on Error Rate")
    
    # Compare number of neurons per layer
    print("\n=== EXPERIMENT 2: COMPARING NUMBER OF NEURONS PER LAYER ===")
    neuron_counts, pretrained_neuron_errors, random_neuron_errors = compare_neuron_count(
        X_train, y_train_onehot, X_test, y_test_onehot)
    
    plot_performance_comparison(
        neuron_counts, pretrained_neuron_errors, random_neuron_errors,
        "Number of Neurons per Layer", "Error Rate",
        "Effect of Layer Width on Error Rate")
    
    # Compare training set size
    print("\n=== EXPERIMENT 3: COMPARING TRAINING SET SIZE ===")
    sample_sizes, pretrained_size_errors, random_size_errors = compare_training_size(
        X_train, y_train, y_train_onehot, X_test, y_test_onehot)
    
    plot_performance_comparison(
        sample_sizes, pretrained_size_errors, random_size_errors,
        "Training Set Size", "Error Rate",
        "Effect of Training Set Size on Error Rate")
    
    # Find the best configuration and visualize output probabilities
    print("\n=== FINDING OPTIMAL CONFIGURATION ===")
    
    # Based on experiment results, choose best parameters
    best_layers = 2  # Example - replace with actual best value from experiments
    best_neurons = 500  # Example - replace with actual best value from experiments
    
    # Train models with optimal configuration
    layer_sizes = [784] + [best_neurons] * best_layers + [10]
    
    print("\nTraining DNN with optimal configuration and pre-training...")
    dnn_pretrained, _, _ = train_and_evaluate(
        X_train, y_train_onehot, X_test, y_test_onehot,
        layer_sizes, use_pretraining=True, verbose=True)
    
    print("\nShowing output probabilities for a few test samples...")
    show_output_probabilities(dnn_pretrained, X_test, y_test)
    
    print("\nExperiments completed. Results saved as PNG files.")

if __name__ == "__main__":
    run_experiments()