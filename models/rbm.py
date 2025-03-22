import numpy as np
from scipy.special import expit  # sigmoid function

class RBM:
    def __init__(self, n_visible, n_hidden):
        """
        Initialize a Restricted Boltzmann Machine.
        
        Parameters:
        -----------
        n_visible: int
            Number of visible units
        n_hidden: int
            Number of hidden units
        """
        # Set number of visible and hidden units
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Initialize weights and biases
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)
        self.b = np.zeros(n_hidden)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        #return 1 / (1 + np.exp(-x))
        return expit(x)
    
    def sample_hidden(self, v):
        """
        Compute probabilities and samples of hidden units.
        
        Parameters:
        -----------
        v: array-like
            Visible units
            
        Returns:
        --------
        tuple
            (p(h|v), sampled h)
        """
        # Calculate p(h|v)
        p_h = self.sigmoid(v @ self.W + self.b)
        # Sample h from p(h|v)
        h = (np.random.random(p_h.shape) < p_h).astype(np.float64)
        return p_h, h
    
    def sample_visible(self, h):
        """
        Compute probabilities and samples of visible units.
        
        Parameters:
        -----------
        h: array-like
            Hidden units
            
        Returns:
        --------
        tuple
            (p(v|h), sampled v)
        """
        # Calculate p(v|h)
        p_v = self.sigmoid(np.dot(h, self.W.T) + self.a)
        # Sample v from p(v|h)
        v = (np.random.random(p_v.shape) < p_v).astype(np.float64)
        return p_v, v
    
    def contrastive_divergence(self, v0):
        """
        Contrastive Divergence algorithm for training RBM.
        
        Parameters:
        -----------
        v0: array-like
            Positive phase visible units (input data)
        """
        #First phase
        p_h0, _ = self.sample_hidden(v0)

        # Gibbs sampling
        vk = v0.copy()
        for _ in range(self.k):
            # Positive phase
            _, hk = self.sample_hidden(vk)
            # Negative phase
            _, vk = self.sample_visible(hk)
        
        # Last phase
        p_hk, _ = self.sample_hidden(vk)
        
        # Parameter updates (gradient ascent)
        grad_a = np.mean(v0 - vk, axis=0)
        grad_b = np.mean(p_h0 - p_hk, axis=0)
        grad_W = (v0.T @ p_h0 - vk.T @ p_hk) / v0.shape[0]
        
        # Weight updates
        self.W += self.lr * grad_W
        
        # Bias updates
        self.a += self.lr * grad_a
        self.b += self.lr * grad_b
        
        # Compute reconstruction error
        reconstruction_error = np.mean((v0 - vk) ** 2)
        return reconstruction_error
    
    def fit(self,
            data,
            batch_size=100,
            nb_epochs=100,
            k=1,
            lr=0.1,
            decay_rate=1.0,
            verbose=True):
        """
        Train the RBM using Contrastive Divergence.
        
        Parameters:
        -----------
        data: array-like
            Training data
        batch_size: int
            Batch size
        nb_epochs: int
            Number of epochs
        k: int
            Number of Gibbs sampling steps
        lr: float
            Initial learning rate
        decay_rate: float
            Learning rate decay rate
        verbose: bool
            Whether to print progress
            
        Returns:
        --------
        self
        """
        # Set data size
        self.data_size = len(data)
        
        # Set hyperparameters
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.k = k
        self.lr = lr
        self.decay_rate = decay_rate
        
        # List to store training errors
        self.errors = []
        
        # Train the RBM
        for epoch in range(self.nb_epochs):
            # Shuffle data
            indices = np.random.permutation(self.data_size)
            data_shuffled = data[indices]
            
            # Initialize epoch error
            epoch_error = 0
            batch_count = 0
            
            # Split data into batches
            batches = np.array_split(data, max(1, self.data_size // self.batch_size))
            
            # Train on batches
            for batch in batches:
                batch_error = self.contrastive_divergence(batch)
                epoch_error += batch_error
                batch_count += 1
            
            # Compute average error
            epoch_error /= batch_count
            self.errors.append(epoch_error)
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch + 1}/{self.nb_epochs}: Error {epoch_error:.4f}")
            
            # Update learning rate
            self.lr = self.lr * self.decay_rate
            
        return self.errors
    
    def generate_samples(self, n_samples=10, gibbs_steps=200):
        """
        Generate samples from the RBM.
        
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
        # Start with random visible units
        samples = np.random.binomial(1, 0.5, (n_samples, self.n_visible))
        
        # Perform Gibbs sampling
        for _ in range(gibbs_steps):
            _, h = self.sample_hidden(samples)
            _ , samples = self.sample_visible(h)
            
        return samples
