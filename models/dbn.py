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
        # Store layer sizes
        self.layer_sizes = layer_sizes
        
        # Create RBMs for each pair of layers
        self.rbms = [
            RBM(layer_sizes[i], layer_sizes[i+1]) 
            for i in range(len(layer_sizes)-1)
        ]

    def fit(self,
            data,
            batch_size=100,
            nb_epochs=100,
            k=1,
            lr=0.1,
            decay_rate=1.0,
            verbose=True):
        """
        Train the DBN using greedy layer-wise pretraining.
        
        Parameters:
        -----------
        data: array-like
            Training data
        batch_size: int
            Batch size
        nb_epochs: int
            Number of epochs per layer
        k: int
            Number of Gibbs sampling steps
        lr: float
            Learning rate
        decay_rate: float
            Learning rate decay rate
        verbose: bool
            Whether to print progress
        
        Returns:
        --------
        self
        """
        # Set hyperparameters
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.k = k
        self.lr = lr
        self.decay_rate = decay_rate
        
        # List to store pretraining errors
        self.pretrain_errors = []
        
        input_data = data.copy()
        
        for i, rbm in enumerate(self.rbms):
            if verbose:
                print(f"Pretraining layer {i+1}/{len(self.rbms)}...")

            rbm.fit(input_data,
                    self.batch_size,
                    self.nb_epochs,
                    self.k,
                    self.lr,
                    self.decay_rate,
                    verbose)
            
            self.pretrain_errors.append(rbm.errors)
            
            if i < len(self.rbms) - 1:
                input_data = rbm.transform(input_data)
        
        return self.pretrain_errors
    
    def generate_samples(self, n_samples=10, gibbs_steps=200):
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
        # Initialize with random samples from the top layer
        h_samples = np.random.binomial(1, 0.5, (n_samples, self.layer_sizes[-1]))
        
        # Get the top-level RBM
        top_rbm = self.rbms[-1]
        
        # Perform Gibbs sampling at the top level
        for _ in range(gibbs_steps):
            _, v_samples = top_rbm.sample_visible(h_samples)
            _, h_samples = top_rbm.sample_hidden(v_samples)
        
        # Start with visible samples from top RBM
        h_samples = v_samples
        
        # Propagate down through the layers
        for i in range(len(self.rbms) - 2, -1, -1):
            _, v_samples = self.rbms[i].sample_visible(h_samples)
            h_samples = v_samples
        
        # Return the visible samples from the bottom RBM
        return v_samples