import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import log_loss

class DNN:
    def __init__(self, layer_sizes, dbn=None):
        """
        Initialize a Deep Neural Network, optionally with weights from a DBN.
        
        Parameters:
        -----------
        layer_sizes: list
            List of layer sizes [input_size, hidden1_size, ..., output_size]
        dbn: DBN, optional
            A trained Deep Belief Network for weight initialization
        """
        # Store layer sizes
        self.layer_sizes = layer_sizes
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []

        if dbn:
            # Weight and bias initialization from DBN
            for rbm in dbn.rbms:
                self.weights.append(rbm.W.copy())
                self.biases.append(rbm.b.copy())
            
            # Add final layer weights and biases
            self.weights.append(np.random.randn(layer_sizes[-2], layer_sizes[-1]) * 0.01)
            self.biases.append(np.zeros(layer_sizes[-1]))
            
            # Store pretraining errors
            self.pretrain_errors = dbn.pretrain_errors
            
        else:
            # Random initialization
            for i in range(len(layer_sizes) - 1):
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
                self.biases.append(np.zeros(layer_sizes[i + 1]))

    def softmax(self, z):
        """Softmax activation function with numerical stability."""
        #exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        #return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return softmax(z, axis=1)

    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability."""
        #return 1 / (1 + np.exp(-z))
        return expit(z)

    def cross_entropy_loss(self, y_true, y_pred):
        """Cross-entropy loss function."""
        #return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        return log_loss(y_true, y_pred)

    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X: array-like
            Input data
            
        Returns:
        --------
        list
            List of activations at each layer
        """
        activations = [X]
        
        # Pass through hidden layers
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = self.sigmoid(X @ W + b)
            activations.append(X)
        
        # Logits for final layer
        logits = X @ self.weights[-1] + self.biases[-1]
        activations.append(self.softmax(logits))
        return activations

    def backward(self, activations, y_true):
        """
        Backward pass for gradient computation and weight updates.
        
        Parameters:
        -----------
        activations: list
            Activations from forward pass
        y_true: array-like
            True labels (one-hot encoded)
        """
        deltas = [activations[-1] - y_true]

        # Compute deltas for hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            delta = deltas[0] @ self.weights[i].T * activations[i] * (1 - activations[i])
            deltas.insert(0, delta)

        # Update weights and biases
        for i in range(len(self.weights)):
            grad_W = activations[i].T @ deltas[i] / y_true.shape[0]
            grad_b = np.mean(deltas[i], axis=0)
            
            self.weights[i] -= self.lr * grad_W
            self.biases[i] -= self.lr * grad_b

    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            batch_size=100,
            nb_epochs=200,
            lr=0.1,
            decay_rate=1.0,
            verbose=True):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X_train: array-like
            Training data
        y_train: array-like
            Training labels (one-hot encoded)
        X_val: array-like, optional
            Validation data
        y_val: array-like, optional
            Validation labels (one-hot encoded)
        batch_size: int
            Batch size
        nb_epochs: int
            Number of epochs
        lr: float
            Initial learning rate
        decay_rate: float
            Learning rate decay rate
        verbose: bool
            Whether to print progress
            
        Returns:
        --------
        list
            Training errors
        """
        # Set hyperparameters
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.decay_rate = decay_rate
        
        # List to store errors
        self.errors = []
        
        # Training the DNN
        for epoch in range(nb_epochs):
            # Learning rate schedule
            self.lr = self.lr * self.decay_rate
            
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Divide into mini-batches
            X_batches = np.array_split(X_train_shuffled, range(batch_size, X_train.shape[0], batch_size))
            y_batches = np.array_split(y_train_shuffled, range(batch_size, y_train.shape[0], batch_size))
            
            # Train on mini-batches
            for X_batch, y_batch in zip(X_batches, y_batches):
                activations = self.forward(X_batch)
                self.backward(activations, y_batch)

            # Compute error
            train_error = self.cross_entropy_loss(y_train, self.forward(X_train)[-1])

            # Compute validation error
            if X_val is not None and y_val is not None:
                val_error = self.cross_entropy_loss(y_val, self.forward(X_val)[-1])
                if verbose:
                    print(f"Epoch {epoch+1}/{nb_epochs} - Train error: {train_error:.4f} - Val error: {val_error:.4f}")
                self.errors.append((train_error, val_error))
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{nb_epochs} - Train Loss: {train_error:.4f}")
                self.errors.append(train_error)

        return self.errors

    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X: array-like
            Input data
            
        Returns:
        --------
        array-like
            Predicted class indices
        """
        probs = self.forward(X)[-1]
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_true):
        """
        Calculate accuracy.
        
        Parameters:
        -----------
        X: array-like
            Input data
        y_true: array-like
            True labels (one-hot encoded)
            
        Returns:
        --------
        float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == np.argmax(y_true, axis=1))
    
    def error_rate(self, X, y_true):
        """
        Calculate error rate.
        
        Parameters:
        -----------
        X: array-like
            Input data
        y_true: array-like
            True labels (one-hot encoded)
            
        Returns:
        --------
        float
            Error rate
        """
        return 1 - self.accuracy(X, y_true)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X: array-like
            Input data
            
        Returns:
        --------
        array-like
            Predicted class probabilities
        """
        return self.forward(X)[-1]