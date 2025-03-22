import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import Binarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import pickle

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

#===============================================================================
# Dataset Loading Functions
#===============================================================================

def load_binary_alphadigits(chars=None, return_labels=False):
    """
    Load Binary AlphaDigits dataset.
    
    Parameters:
    -----------
    chars: list or None
        List of characters to load (0-9 for digits, 10-35 for uppercase letters)
        If None, load all characters
    return_labels: bool
        Whether to return character labels with the data
        
    Returns:
    --------
    data: array-like
        Binary images with shape (n_samples, 20*16)
    labels: array-like (optional)
        Character labels corresponding to each sample
    """
    try:
        mat = sio.loadmat('/Users/christopher/Library/CloudStorage/OneDrive-ENSTAParis/ENSTA/3A/Deep Learning II/deep_learning_2/data/binaryalphadigs.mat')
        digits = mat['dat']
        
        if chars is None:
            chars = list(range(36))
        
        samples = []
        labels = []
        
        for char_idx in chars:
            for sample in digits[char_idx]:
                samples.append(sample.flatten())
                labels.append(char_idx)
                
        # Convert labels to character representation
        char_labels = []
        for label in labels:
            if label < 10:  # Digits 0-9
                char_labels.append(str(label))
            else:  # Letters A-Z (10-35)
                char_labels.append(chr(ord('A') + label - 10))
        
        if return_labels:
            return np.array(samples), np.array(char_labels)
        else:
            return np.array(samples)
    
    except:
        print("Error: Could not load Binary AlphaDigits dataset.")
        print("Please download it from https://www.kaggle.com/datasets/angevalli/binary-alpha-digits")
        print("and place it in /Users/christopher/Library/CloudStorage/OneDrive-ENSTAParis/ENSTA/3A/Deep Learning II/project/data/")
        return (None, None) if return_labels else None

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
