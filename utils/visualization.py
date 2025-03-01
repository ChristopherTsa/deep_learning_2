import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss", save_path=None):
    """Plot a single loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        
    plt.show()

def plot_multiple_curves(curves_dict, title="Loss Comparison", xlabel="Epoch", ylabel="Loss", save_path=None):
    """
    Plot multiple curves on the same graph.
    
    Parameters:
    -----------
    curves_dict: dict
        Dictionary mapping curve names to y-values
    save_path: str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 7))
    
    for name, values in curves_dict.items():
        plt.plot(values, label=name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    
    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        
    plt.show()

def plot_performance_comparison(x_values, pretrained_errors, random_errors, 
                              xlabel, ylabel, title, save_path=None):
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
    save_path: str, optional
        Path to save the figure
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
    
    # Use provided save_path or default based on title
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    elif title:
        file_path = f"{title.replace(' ', '_')}.png"
        plt.savefig(file_path)
        print(f"Figure saved to {file_path}")
    
    plt.show()

def display_binary_images(images, n_cols=10, figsize=(10, 10), titles=None, save_path=None):
    """
    Display binary images in a grid.
    
    Parameters:
    -----------
    images: array-like
        Binary images with shape (n_samples, height*width)
    n_cols: int
        Number of columns in the grid
    figsize: tuple
        Figure size
    titles: list or None
        Titles for each image
    save_path: str, optional
        Path to save the figure
    """
    n_images = len(images)
    n_rows = (n_images - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_images):
        # For AlphaDigits dataset, images are 20x16
        if images[i].size == 320:  # 20*16 = 320
            img = images[i].reshape(20, 16)
        elif images[i].size == 784:  # 28*28 = 784 (MNIST)
            img = images[i].reshape(28, 28)
        else:
            # Try to make a square image
            side = int(np.sqrt(images[i].size))
            img = images[i].reshape(side, -1)
            
        axes[i].imshow(img, cmap='binary')
        axes[i].axis('off')
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])
            
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    
    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()