import numpy as np
import matplotlib.pyplot as plt

def display_weight_matrices(weights, shape=(28, 28), n_samples=100, n_cols=10):
    """Display the weight matrices as images."""
    n_samples = min(n_samples, weights.shape[0])
    n_rows = (n_samples - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = axes.flatten()
    
    for i in range(n_samples):
        weights_img = weights[i, :].reshape(shape)
        axes[i].imshow(weights_img, cmap='gray')
        axes[i].axis('off')
    
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_reconstructions(original, reconstructed, n_samples=10):
    """Display original and reconstructed images side by side."""
    n_samples = min(n_samples, original.shape[0])
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples, 2))
    
    for i in range(n_samples):
        # Determine shape based on input size
        if original[i].size == 784:  # MNIST
            shape = (28, 28)
        elif original[i].size == 320:  # AlphaDigits
            shape = (20, 16)
        else:
            side = int(np.sqrt(original[i].size))
            shape = (side, side)
            
        # Display original
        axes[0, i].imshow(original[i].reshape(shape), cmap='binary')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
            
        # Display reconstruction
        axes[1, i].imshow(reconstructed[i].reshape(shape), cmap='binary')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.show()

def plot_loss_curve(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss"):
    """Plot a single loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_multiple_curves(curves_dict, title="Loss Comparison", xlabel="Epoch", ylabel="Loss"):
    """
    Plot multiple curves on the same graph.
    
    Parameters:
    -----------
    curves_dict: dict
        Dictionary mapping curve names to y-values
    """
    plt.figure(figsize=(12, 7))
    
    for name, values in curves_dict.items():
        plt.plot(values, label=name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_hidden_activations(activations, n_cols=10):
    """Visualize hidden layer activations."""
    n_samples = activations.shape[0]
    n_units = activations.shape[1]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(activations.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation')
    plt.xlabel('Sample')
    plt.ylabel('Hidden Unit')
    plt.title(f'Hidden Layer Activations ({n_samples} samples, {n_units} units)')
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(x_values, pretrained_errors, random_errors, 
                              xlabel, ylabel, title):
    """
    Plot comparison between pretrained and randomly initialized network performance.
    
    Parameters:
    -----------
    x_values: list
        X-axis values (e.g., number of layers, neurons, or training samples)
    pretrained_errors: list
        Error rates for pretrained networks
    random_errors: list
        Error rates for randomly initialized networks
    xlabel: str
        X-axis label
    ylabel: str
        Y-axis label
    title: str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(x_values, pretrained_errors, 'b-o', label='Pre-trained Network')
    plt.plot(x_values, random_errors, 'r-s', label='Random Initialization')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Add data value labels
    for i, (x, y1, y2) in enumerate(zip(x_values, pretrained_errors, random_errors)):
        plt.annotate(f'{y1:.4f}', (x, y1), textcoords="offset points", 
                     xytext=(0,10), ha='center')
        plt.annotate(f'{y2:.4f}', (x, y2), textcoords="offset points", 
                     xytext=(0,-15), ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def plot_learning_curves(histories, labels, title="Learning Curves"):
    """
    Plot learning curves from training histories.
    
    Parameters:
    -----------
    histories: list of lists
        List of training histories, each containing loss values
    labels: list
        List of labels for each history
    title: str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for history, label in zip(histories, labels):
        plt.plot(history, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()