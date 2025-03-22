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

    def reconstruction_loss(self, v0, vk):
        """
        Compute the reconstruction loss.
        
        Parameters:
        -----------
        v0: array-like
            Input data
        vk: array-like
            Reconstructed data
            
        Returns:
        --------
        float
            Reconstruction loss
        """
        return np.mean((v0 - vk) ** 2)
    
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
            
        Returns:
        --------
        float
            Reconstruction loss
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
        
        # Compute and return reconstruction loss
        return self.reconstruction_loss(v0, vk)
    
    def fit(self,
            data,
            validation_data=None,
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
        validation_data: array-like, optional
            Validation data for monitoring overfitting
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
        
        # List to store training and validation losses
        self.losses = []
        self.val_losses = [] if validation_data is not None else None
        
        # Train the RBM
        for epoch in range(self.nb_epochs):
            # Shuffle data
            indices = np.random.permutation(self.data_size)
            data_shuffled = data[indices]
            
            # Initialize epoch loss
            epoch_loss = 0
            batch_count = 0
            
            # Split data into batches
            batches = np.array_split(data_shuffled, max(1, self.data_size // self.batch_size))
            
            # Train on batches
            for batch in batches:
                batch_loss = self.contrastive_divergence(batch)
                epoch_loss += batch_loss
                batch_count += 1
            
            # Compute average loss
            epoch_loss /= batch_count
            self.losses.append(epoch_loss)
            
            # Compute validation loss if validation data is provided
            if validation_data is not None:
                # For validation, we only need the reconstruction loss, not parameter updates
                val_batches = np.array_split(validation_data, max(1, len(validation_data) // self.batch_size))
                val_loss = 0
                val_batch_count = 0
                
                for val_batch in val_batches:
                    # Get reconstructions
                    vk = self.reconstruct(val_batch, gibbs_steps=k)
                    # Compute and accumulate loss
                    val_loss += self.reconstruction_loss(val_batch, vk)
                    val_batch_count += 1
                
                # Compute average validation loss
                val_loss /= val_batch_count
                self.val_losses.append(val_loss)
                
                # Print progress with validation loss
                if verbose:
                    print(f"Epoch {epoch + 1}/{self.nb_epochs}: Train Loss {epoch_loss:.4f}, Val Loss {val_loss:.4f}")
            else:
                # Print progress without validation loss
                if verbose:
                    print(f"Epoch {epoch + 1}/{self.nb_epochs}: Loss {epoch_loss:.4f}")
            
            # Update learning rate
            self.lr = self.lr * self.decay_rate
        
        return self
    
    def transform(self, data):
        """
        Transform data to hidden representation.
        
        Parameters:
        -----------
        data: array-like
            Input data
            
        Returns:
        --------
        array-like
            Hidden representation
        """
        p_h, _ = self.sample_hidden(data)
        return p_h
    
    def reconstruct(self, data, gibbs_steps=1):
        """
        Reconstruct data
        
        Parameters:
        -----------
        data: array-like
            Input data to reconstruct
        gibbs_steps: int
            Number of Gibbs sampling steps
            
        Returns:
        --------
        array-like
            Reconstructed data
        """
        v = data.copy()
        for _ in range(gibbs_steps):
            _, h = self.sample_hidden(v)
            _, v = self.sample_visible(h)
        
        return v
    
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
        samples = self.reconstruct(samples, gibbs_steps)
            
        return samples
