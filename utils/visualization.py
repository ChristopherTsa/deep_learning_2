import numpy as np
import matplotlib.pyplot as plt
import math

#===============================================================================
# Plotting Functions
#===============================================================================

def plot_losses(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss", 
               layer_names=None, save_path=None):
    """
    Plot one or more loss curves.
    
    Parameters:
    -----------
    losses: list or list of lists
        Single loss curve or multiple loss curves (e.g., one per layer)
    title: str
        Plot title
    xlabel: str
        X-axis label
    ylabel: str
        Y-axis label
    layer_names: list of str, optional
        Names for each loss curve (e.g., 'Layer 1', 'Layer 2')
    save_path: str, optional
        Path to save the figure
        
    Returns:
    --------
    fig: matplotlib Figure object
        The created figure (for further manipulation if needed)
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Check if losses is a list of lists (multiple curves)
    if losses and isinstance(losses[0], (list, np.ndarray)):
        # Multiple loss curves
        for i, curve in enumerate(losses):
            label = layer_names[i] if layer_names and i < len(layer_names) else f'Layer {i+1}'
            plt.plot(curve, label=label)
        plt.legend()
    else:
        # Single loss curve
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
    
    return fig

def plot_comparison(x_values, pretrained_test_errors, random_test_errors, 
                   pretrained_train_errors=None, random_train_errors=None,
                   xlabel="", ylabel="", title="", legend_labels=None, 
                   dual_plot=True, save_path=None):
    """
    Plot comparison between pretrained and randomly initialized network performance.
    
    Parameters:
    -----------
    x_values: list
        X-axis values (e.g., number of layers, neurons, or training samples)
    pretrained_test_errors: list
        Test error rates for pretrained networks
    random_test_errors: list
        Test error rates for randomly initialized networks
    pretrained_train_errors: list, optional
        Train error rates for pretrained networks
    random_train_errors: list, optional
        Train error rates for randomly initialized networks
    xlabel: str
        X-axis label
    ylabel: str
        Y-axis label
    title: str
        Plot title
    legend_labels: list, optional
        Custom labels for the legend
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
    
    # Use consistent colors for each initialization method
    pretrained_color = 'blue'
    random_color = 'red'
    
    # Plot test errors with solid lines
    plt.plot(x_values, pretrained_test_errors, color=pretrained_color, marker='o', linestyle='-', 
             label=legend_labels[0] if legend_labels else 'Pre-trained (Test)')
    plt.plot(x_values, random_test_errors, color=random_color, marker='s', linestyle='-', 
             label=legend_labels[1] if legend_labels else 'Random Init (Test)')
    
    # Plot train errors with dashed lines (if provided)
    if pretrained_train_errors is not None:
        plt.plot(x_values, pretrained_train_errors, color=pretrained_color, marker='o', linestyle='--', 
                 label=legend_labels[2] if legend_labels and len(legend_labels) > 2 else 'Pre-trained (Train)')
    
    if random_train_errors is not None:
        plt.plot(x_values, random_train_errors, color=random_color, marker='s', linestyle='--', 
                 label=legend_labels[3] if legend_labels and len(legend_labels) > 3 else 'Random Init (Train)')
    
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
        
        # Plot test errors
        plt.plot(x_values, pretrained_test_errors, color=pretrained_color, linestyle='-', marker='o', 
                 label=legend_labels[0] if legend_labels else 'Pre-trained (Test)')
        plt.plot(x_values, random_test_errors, color=random_color, linestyle='-', marker='s', 
                 label=legend_labels[1] if legend_labels else 'Random Init (Test)')
        
        # Plot train errors if provided
        if pretrained_train_errors is not None:
            plt.plot(x_values, pretrained_train_errors, color=pretrained_color, linestyle='--', marker='o', 
                     label=legend_labels[2] if legend_labels and len(legend_labels) > 2 else 'Pre-trained (Train)')
        
        if random_train_errors is not None:
            plt.plot(x_values, random_train_errors, color=random_color, linestyle='--', marker='s', 
                     label=legend_labels[3] if legend_labels and len(legend_labels) > 3 else 'Random Init (Train)')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title} (Detailed Comparison)")
        plt.grid(True)
        plt.legend()
        
        # Add data value labels for test errors
        for i, (x, y1, y2) in enumerate(zip(x_values, pretrained_test_errors, random_test_errors)):
            plt.annotate(f'{y1:.4f}', (x, y1), textcoords="offset points", 
                        xytext=(0,10), ha='center')
            plt.annotate(f'{y2:.4f}', (x, y2), textcoords="offset points", 
                        xytext=(0,-15), ha='center')
        
        # Add data value labels for train errors (if provided)
        if pretrained_train_errors is not None and random_train_errors is not None:
            for i, (x, y1, y2) in enumerate(zip(x_values, pretrained_train_errors, random_train_errors)):
                plt.annotate(f'{y1:.4f}', (x, y1), textcoords="offset points", 
                            xytext=(0,10), ha='center')
                plt.annotate(f'{y2:.4f}', (x, y2), textcoords="offset points", 
                            xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        
        # Save detail plot
        detail_plot_path = f"{save_path}_detail.png" if save_path else f"{title.replace(' ', '_')}_detail.png"
        plt.savefig(detail_plot_path)
        saved_files['detail_plot'] = detail_plot_path
        plt.show()
    
    print(f"Plots for {title} saved successfully.")
    return saved_files

#===============================================================================
# Visualization Functions
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

def display_weights(rbm, height=20, width=16, figsize=(10, 10), n_cols=10, save_path=None):
    """
    Plot the RBM weights as images.
    
    Parameters:
    -----------
    rbm: RBM
        A trained RBM model
    height: int
        Height of the weight images
    width: int
        Width of the weight images
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
