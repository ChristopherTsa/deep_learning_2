import numpy as np
import matplotlib.pyplot as plt
from models import RBM

class DBN:
    def __init__(self, layer_sizes):
        """
        Initialize a Deep Belief Network.
        
        Parameters:
        -----------
        layer_sizes: list
            List of layer sizes [visible_size, hidden1_size, ..., hiddenN_size]
        """
        self.layer_sizes = layer_sizes
        self.rbms = [
            RBM(layer_sizes[i], layer_sizes[i+1]) 
            for i in range(len(layer_sizes)-1)
        ]
        self.trained = False
        self.pretrain_errors = []

    def fit(self, data, nb_epochs=10, batch_size=100, lr=0.01, k=1,
            momentum=0.0, weight_decay=0.0, l1_reg=0.0, decay_rate=1.0, verbose=True):
        """
        Train the DBN using greedy layer-wise pretraining.
        
        Parameters:
        -----------
        data: array-like
            Training data
        nb_epochs: int
            Number of epochs per layer
        batch_size: int
            Batch size
        lr: float
            Learning rate
        k: int
            Number of Gibbs sampling steps
        momentum: float
            Momentum coefficient
        weight_decay: float
            L2 regularization parameter
        l1_reg: float
            L1 regularization parameter
        decay_rate: float
            Learning rate decay rate
        verbose: bool
            Whether to print progress
            
        Returns:
        --------
        self
        """
        self.trained = True
        input_data = data.copy()
        for i, rbm in enumerate(self.rbms):
            print(f"Pretraining layer {i+1}/{len(self.rbms)}...")
            rbm.fit(input_data, nb_epochs, batch_size, lr, k,
                    momentum, weight_decay, l1_reg, decay_rate, verbose)
            self.pretrain_errors.append(rbm.losses)
            p_h, _ = rbm.sample_hidden(input_data)
            input_data = p_h
        return self
    
    def transform(self, X):
        """
        Transform data through all layers of the DBN.
        
        Parameters:
        -----------
        X: array-like
            Input data
            
        Returns:
        --------
        array-like
            Transformed data (top-level representation)
        """
        hidden = X.copy()
        for rbm in self.rbms:
            hidden = rbm.transform(hidden)
        return hidden

    def predict(self, X):
        """
        Reconstruct input data by going up through the network and back down.
        
        Parameters:
        -----------
        X: array-like
            Input data
            
        Returns:
        --------
        array-like
            Reconstructed data
        """
        # Go up through layers
        hidden = X.copy()
        for rbm in self.rbms:
            hidden = rbm.transform(hidden)
        
        # Come back down through layers
        for i in reversed(range(len(self.rbms))):
            p_v, _ = self.rbms[i].sample_visible(hidden)
            hidden = p_v
        return hidden
    
    def evaluate(self, X):
        """
        Evaluate the DBN by computing reconstruction error.
        
        Parameters:
        -----------
        X: array-like
            Test data
            
        Returns:
        --------
        float
            Mean squared reconstruction error
        """
        reconstructions = self.predict(X)
        return np.mean((X - reconstructions) ** 2)
    
    def plot_pretraining_errors(self):
        """Plot pretraining errors for each layer."""
        plt.figure(figsize=(10, 6))
        for i, errors in enumerate(self.pretrain_errors):
            plt.plot(errors, label=f'Layer {i+1}')
        plt.title("Pretraining errors by layer")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Error")
        plt.legend()
        plt.show()

    def sample(self, n_samples=10, gibbs_steps=1000):
        """
        Generate samples from the DBN.
        
        Parameters:
        -----------
        n_samples: int
            Number of samples to generate
        gibbs_steps: int
            Number of Gibbs sampling steps
            
        Returns:
        --------
        array-like
            Generated samples
        """
        # Start with random visible units at the top layer
        top_samples = np.random.binomial(1, 0.5, (n_samples, self.layer_sizes[-1]))
        
        # Propagate down through the layers
        samples = top_samples
        for i in reversed(range(len(self.rbms))):
            # Initialize with samples from the layer above
            h = samples
            
            # Perform Gibbs sampling at this layer
            for _ in range(gibbs_steps):
                p_v, v = self.rbms[i].sample_visible(h)
                p_h, h = self.rbms[i].sample_hidden(v)
            
            # Keep visible samples for the next layer down
            p_v, samples = self.rbms[i].sample_visible(h)
        
        # Return the visible samples from the bottom RBM
        return samples