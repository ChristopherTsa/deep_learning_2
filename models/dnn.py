import numpy as np

def xavier_init(fan_in, fan_out):
    """Xavier weight initialization."""
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

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
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        if dbn:
            # Initialize weights and biases from the DBN
            for rbm in dbn.rbms:
                self.weights.append(rbm.W.copy())
                self.biases.append(rbm.b.copy())
            # Add the final layer for classification
            self.weights.append(xavier_init(layer_sizes[-2], layer_sizes[-1]))
            self.biases.append(np.zeros(layer_sizes[-1]))
        else:
            # Random initialization if no DBN provided
            for i in range(len(layer_sizes) - 1):
                self.weights.append(xavier_init(layer_sizes[i], layer_sizes[i + 1]))
                self.biases.append(np.zeros(layer_sizes[i + 1]))

    def softmax(self, z):
        """Softmax activation function with numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, y_true, y_pred):
        """Cross-entropy loss function."""
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

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
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = self.sigmoid(np.dot(X, W) + b)
            activations.append(X)
        
        # Last layer with softmax
        logits = np.dot(X, self.weights[-1]) + self.biases[-1]
        activations.append(self.softmax(logits))
        return activations

    def backward(self, activations, y_true, lr, reg_lambda=0.0):
        """
        Backward pass for gradient computation and weight updates.
        
        Parameters:
        -----------
        activations: list
            Activations from forward pass
        y_true: array-like
            True labels (one-hot encoded)
        lr: float
            Learning rate
        reg_lambda: float
            L2 regularization parameter
        """
        deltas = [activations[-1] - y_true]

        # Calculate gradients for hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * activations[i] * (1 - activations[i])
            deltas.insert(0, delta)

        # Update weights and biases
        for i in range(len(self.weights)):
            grad_W = np.dot(activations[i].T, deltas[i]) / y_true.shape[0]
            grad_b = np.mean(deltas[i], axis=0)
            
            # Add L2 regularization penalty
            grad_W += reg_lambda * self.weights[i]
            
            # Update weights and biases
            self.weights[i] -= lr * grad_W
            self.biases[i] -= lr * grad_b

    def fit(self, X_train, y_train, X_val=None, y_val=None, nb_epochs=10, batch_size=100,
            lr=0.01, decay_rate=1.0, reg_lambda=0.0, verbose=True):
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
        nb_epochs: int
            Number of epochs
        batch_size: int
            Batch size
        lr: float
            Initial learning rate
        decay_rate: float
            Learning rate decay rate
        reg_lambda: float
            L2 regularization parameter
        verbose: bool
            Whether to print progress
            
        Returns:
        --------
        list
            Training history
        """
        history = []
        initial_lr = lr
        for epoch in range(nb_epochs):
            
            # Apply learning rate decay
            if epoch > 20:
                lr = initial_lr * (decay_rate ** epoch)
            
            # Shuffle data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Split into mini-batches
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                activations = self.forward(X_batch)
                self.backward(activations, y_batch, lr, reg_lambda)

            # Calculate loss on training set
            train_loss = self.cross_entropy_loss(y_train, self.forward(X_train)[-1])

            if X_val is not None and y_val is not None:
                val_loss = self.cross_entropy_loss(y_val, self.forward(X_val)[-1])
                if verbose:
                    print(f"Epoch {epoch+1}/{nb_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                history.append((train_loss, val_loss))
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{nb_epochs} - Train Loss: {train_loss:.4f}")
                history.append(train_loss)

        return history

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