import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import Binarizer, OneHotEncoder
import scipy.io as sio
import matplotlib.pyplot as plt

def load_mnist(binarize_threshold=0.5, normalize=True):
    """
    Load MNIST dataset.
    
    Parameters:
    -----------
    binarize_threshold: float
        Threshold for binarizing images
    normalize: bool
        Whether to normalize pixel values to [0,1]
        
    Returns:
    --------
    X_train: array-like
        Training images
    y_train: array-like
        Training labels
    X_test: array-like
        Test images
    y_test: array-like
        Test labels
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    # Convert to float and normalize if requested
    X = X.astype(float)
    if normalize:
        X = X / 255.0
    
    # Binarize if threshold is provided
    if binarize_threshold is not None:
        X = Binarizer(threshold=binarize_threshold).fit_transform(X)
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Split into train and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    print(f"MNIST loaded: {X_train.shape[0]} training and {X_test.shape[0]} test samples")
    return X_train, y_train, X_test, y_test

def one_hot_encode(y):
    """
    One-hot encode labels.
    
    Parameters:
    -----------
    y: array-like
        Labels
        
    Returns:
    --------
    array-like
        One-hot encoded labels
    """
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(y.reshape(-1, 1))

def load_binary_alphadigits(chars=None):
    """
    Load Binary AlphaDigits dataset.
    
    Parameters:
    -----------
    chars: list or None
        List of characters to load (0-9 for digits, 10-35 for uppercase letters)
        If None, load all characters
        
    Returns:
    --------
    data: array-like
        Binary images with shape (n_samples, 20*16)
    """
    try:
        mat = sio.loadmat('/Users/christopher/Library/CloudStorage/OneDrive-ENSTAParis/ENSTA/3A/Deep Learning II/deep_learning_2/data/binaryalphadigs.mat')
        digits = mat['dat']
        
        if chars is None:
            chars = list(range(36))
        
        samples = []
        
        for char_idx in chars:
            for sample in digits[char_idx]:
                samples.append(sample.flatten())
                
        return np.array(samples)
    
    except:
        print("Error: Could not load Binary AlphaDigits dataset.")
        print("Please download it from https://www.kaggle.com/datasets/angevalli/binary-alpha-digits")
        print("and place it in /Users/christopher/Library/CloudStorage/OneDrive-ENSTAParis/ENSTA/3A/Deep Learning II/project/data/")
        return None