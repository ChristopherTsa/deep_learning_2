import numpy as np
import matplotlib.pyplot as plt
import math

#===============================================================================
# General Plotting Functions
#===============================================================================

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

def plot_neural_net_comparison(x_values, pretrained_errors, random_errors, 
                              xlabel, ylabel, title, dual_plot=True, save_path=None):
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
    dual_plot: bool, default=True
        If True, generates both a line plot and a bar plot comparison
        If False, generates only the line plot
    save_path: str, optional
        Path to save the figure. If None, will generate a path based on title
        
    Returns:
    --------
    dict
        Dictionary with paths to saved figures
    """
    saved_files = {}
    
    # Create the line plot
    plt.figure(figsize=(12, 7))
    
    curves_dict = {
        'Pre-trained': pretrained_errors,
        'Random initialization': random_errors
    }
    
    for name, values in curves_dict.items():
        plt.plot(x_values, values, marker='o', label=name)
    
    plt.title(f"{title} (Line Plot)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_values)
    plt.grid(True)
    plt.legend()
    
    # Save line plot
    line_plot_path = f"{save_path}_line.png" if save_path else f"{title.replace(' ', '_')}_line.png"
    plt.savefig(line_plot_path)
    saved_files['line_plot'] = line_plot_path
    plt.show()
    
    # Create the bar plot with annotations if dual_plot is True
    if dual_plot:
        plt.figure(figsize=(12, 8))
        
        plt.plot(x_values, pretrained_errors, 'b-o', label='Pre-trained Network')
        plt.plot(x_values, random_errors, 'r-s', label='Random Initialization')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title} (Detailed Comparison)")
        plt.grid(True)
        plt.legend()
        
        # Add data value labels
        for i, (x, y1, y2) in enumerate(zip(x_values, pretrained_errors, random_errors)):
            plt.annotate(f'{y1:.4f}', (x, y1), textcoords="offset points", 
                        xytext=(0,10), ha='center')
            plt.annotate(f'{y2:.4f}', (x, y2), textcoords="offset points", 
                        xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        
        # Save bar plot
        bar_plot_path = f"{save_path}_detail.png" if save_path else f"{title.replace(' ', '_')}_detail.png"
        plt.savefig(bar_plot_path)
        saved_files['detail_plot'] = bar_plot_path
        plt.show()
    
    print(f"Plots for {title} saved successfully.")
    return saved_files

#===============================================================================
# Image Visualization Functions
#===============================================================================

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

#===============================================================================
# Model-Specific Visualization Functions
#===============================================================================

def display_rbm_weights(rbm, figsize=(10, 10), n_cols=10, save_path=None):
    """
    Plot the RBM weights as images.
    
    Parameters:
    -----------
    rbm: RBM
        A trained RBM model
    figsize: tuple
        Figure size
    n_cols: int
        Number of columns in the grid
    save_path: str, optional
        Path to save the figure
            
    Returns:
    --------
    fig: matplotlib Figure object
        The created figure (for further manipulation if needed)
    """
    n_vis = rbm.W.shape[0]
    
    # Find appropriate dimensions for visualization
    if math.isqrt(n_vis) ** 2 == n_vis:
        # Perfect square
        side = math.isqrt(n_vis)
        width, height = side, side
        print(f"Weight dimension {n_vis} is a perfect square with side={side}")
    else:
        # Find factors for a rectangular shape
        # Start with the square root and work backwards to find a divisor
        sqrt_n = int(math.sqrt(n_vis))
        
        # Find the first divisor
        for i in range(sqrt_n, 0, -1):
            if n_vis % i == 0:
                width, height = i, n_vis // i
                break
        
        print(f"Weight dimension {n_vis} is not a perfect square. Using dimensions {width}x{height}")
    
    fig, axs = plt.subplots(
        math.ceil(min(rbm.n_hidden, n_cols * n_cols) / n_cols),
        n_cols,
        figsize=figsize
    )
    axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        if i < rbm.n_hidden:
            weight = rbm.W[:, i].reshape(height, width)
            ax.imshow(weight, cmap="gray", interpolation="nearest")
            ax.axis("off")
        else:
            ax.axis("off")
    
    plt.tight_layout()
    
    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig  # Return the figure object for further manipulation

def plot_dbn_pretraining_errors(dbn, save_path=None):
    """
    Plot pretraining errors for each layer of a DBN.
    
    Parameters:
    -----------
    dbn: DBN
        A trained DBN model with pretrain_errors attribute
    save_path: str, optional
        Path to save the figure
            
    Returns:
    --------
    fig: matplotlib Figure object
        The created figure (for further manipulation if needed)
    """
    fig = plt.figure(figsize=(10, 6))
    for i, errors in enumerate(dbn.pretrain_errors):
        plt.plot(errors, label=f'Layer {i+1}')
    plt.title("Pretraining errors by layer")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    
    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig  # Return the figure object for further manipulation