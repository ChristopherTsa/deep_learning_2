import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import Binarizer, OneHotEncoder
import scipy.io as sio
import os
import pickle

#===============================================================================
# Dataset Loading Functions
#===============================================================================

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

def load_mnist(binarize_threshold=0.5, normalize=True, use_cache=True):
    """
    Load MNIST dataset with optional caching. Labels are always one-hot encoded.
    
    Parameters:
    -----------
    binarize_threshold: float
        Threshold for binarizing images
    normalize: bool
        Whether to normalize pixel values to [0,1]
    use_cache: bool
        Whether to use cached version if available
        
    Returns:
    --------
    X_train: array-like
        Training images
    y_train: array-like
        Training labels (one-hot encoded)
    X_test: array-like
        Test images
    y_test: array-like
        Test labels (one-hot encoded)
    """
    mnist_path = "data/mnist.pkl"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Try to load from cache if requested
    if use_cache and os.path.exists(mnist_path):
        print("Loading MNIST dataset from disk cache...")
        with open(mnist_path, 'rb') as f:
            data = pickle.load(f)
            return data['X_train'], data['y_train_onehot'], data['X_test'], data['y_test_onehot']
    
    # Otherwise download the dataset
    print("Downloading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    # Convert to float and normalize if requested
    X = X.astype(float)
    if normalize:
        X = X / 255.0
    
    # Binarize if threshold is provided
    if binarize_threshold is not None:
        X = Binarizer(threshold=binarize_threshold).fit_transform(X)
    
    # Convert labels to integers and ensure they're numpy arrays
    y = y.astype(int)
    
    # Split into train and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Convert pandas Series to numpy arrays if needed
    if hasattr(y_train, 'to_numpy'):
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
    
    # Always one-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    # Save to cache if requested
    if use_cache:
        print("Saving MNIST dataset to disk for future use...")
        data = {
            'X_train': X_train,
            'y_train': y_train,  # Save original labels
            'y_train_onehot': y_train_onehot,  # Save one-hot encoded labels
            'X_test': X_test,
            'y_test': y_test,  # Save original labels
            'y_test_onehot': y_test_onehot  # Save one-hot encoded labels
        }
        with open(mnist_path, 'wb') as f:
            pickle.dump(data, f)
    
    print(f"MNIST loaded: {X_train.shape[0]} training and {X_test.shape[0]} test samples")
    return X_train, y_train_onehot, X_test, y_test_onehot
