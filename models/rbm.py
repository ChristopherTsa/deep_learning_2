import numpy as np

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
        # Initialize weights and biases
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)  # Bias for visible units
        self.b = np.zeros(n_hidden)   # Bias for hidden units
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.losses = []  # To store reconstruction errors during training

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
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
        p_h = self.sigmoid(np.dot(v, self.W) + self.b)
        # Sample h from p(h|v)
        h = np.random.binomial(1, p_h)
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
        v = np.random.binomial(1, p_v)
        return p_v, v
    
    def contrastive_divergence(self, v_pos, k=1, lr=0.01, momentum=0.0, weight_decay=0.0, l1_reg=0.0):
        """
        Contrastive Divergence algorithm for training RBM.
        
        Parameters:
        -----------
        v_pos: array-like
            Positive phase visible units (input data)
        k: int
            Number of Gibbs sampling steps
        lr: float
            Learning rate
        momentum: float
            Momentum coefficient
        weight_decay: float
            L2 regularization parameter
        l1_reg: float
            L1 regularization parameter
        """
        # Initialize momentum updates
        if not hasattr(self, 'dW'):
            self.dW = np.zeros_like(self.W)
            self.da = np.zeros_like(self.a)
            self.db = np.zeros_like(self.b)
        
        # Positive phase
        p_h_pos, h_pos = self.sample_hidden(v_pos)
        
        # Negative phase (start with positive hidden samples)
        h_neg = h_pos.copy()
        for step in range(k):
            p_v_neg, v_neg = self.sample_visible(h_neg)
            p_h_neg, h_neg = self.sample_hidden(v_neg)
        
        # Compute gradients
        batch_size = v_pos.shape[0]
        
        # Positive associations
        pos_associations = np.dot(v_pos.T, p_h_pos)
        # Negative associations
        neg_associations = np.dot(p_v_neg.T, p_h_neg)
        
        # Apply updates with momentum and regularization
        self.dW = momentum * self.dW + lr * ((pos_associations - neg_associations) / batch_size - 
                                            weight_decay * self.W - l1_reg * np.sign(self.W))
        self.da = momentum * self.da + lr * np.mean(v_pos - p_v_neg, axis=0)
        self.db = momentum * self.db + lr * np.mean(p_h_pos - p_h_neg, axis=0)
        
        # Update parameters
        self.W += self.dW
        self.a += self.da
        self.b += self.db
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((v_pos - p_v_neg) ** 2)
        return reconstruction_error
    
    def fit(self, data, nb_epochs=10, batch_size=100, lr=0.01, k=1,
            momentum=0.0, weight_decay=0.0, l1_reg=0.0, decay_rate=1.0, verbose=True,
            momentum_schedule=None):
        """
        Train the RBM using Contrastive Divergence.
        
        Parameters:
        -----------
        data: array-like
            Training data
        nb_epochs: int
            Number of epochs
        batch_size: int
            Batch size
        lr: float
            Initial learning rate
        k: int
            Number of Gibbs sampling steps
        momentum: float
            Initial momentum coefficient
        weight_decay: float
            L2 regularization parameter
        l1_reg: float
            L1 regularization parameter
        decay_rate: float
            Learning rate decay rate
        verbose: bool
            Whether to print progress
        momentum_schedule: dict, optional
            Dictionary with epoch number as key and new momentum value as value
            
        Returns:
        --------
        self
        """
        original_lr = lr
        self.losses = []
        current_momentum = momentum
        
        for epoch in range(nb_epochs):
            # Apply learning rate decay
            if epoch > 0:
                lr = original_lr * (decay_rate ** epoch)
            
            # Apply momentum schedule if provided
            if momentum_schedule is not None and epoch in momentum_schedule:
                current_momentum = momentum_schedule[epoch]
                if verbose:
                    print(f"Epoch {epoch+1}: Updating momentum to {current_momentum}")
            
            # Shuffle data
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data_shuffled = data[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, data.shape[0], batch_size):
                batch = data_shuffled[i:i+batch_size]
                batch_error = self.contrastive_divergence(batch, k, lr, current_momentum, weight_decay, l1_reg)
                epoch_losses.append(batch_error)
            
            avg_loss = np.mean(epoch_losses)
            self.losses.append(avg_loss)
            
            if verbose and (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{nb_epochs} - Reconstruction Error: {avg_loss:.4f}")
        
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
    
    def reconstruct(self, data):
        """
        Reconstruct data.
        
        Parameters:
        -----------
        data: array-like
            Input data
            
        Returns:
        --------
        array-like
            Reconstructed data
        """
        p_h, _ = self.sample_hidden(data)
        p_v, _ = self.sample_visible(p_h)
        return p_v
    
    def generate_samples(self, n_samples=10, gibbs_steps=1000):
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
        v = np.random.binomial(1, 0.5, (n_samples, self.n_visible))
        
        # Perform Gibbs sampling
        for _ in range(gibbs_steps):
            p_h, h = self.sample_hidden(v)
            p_v, v = self.sample_visible(h)
            
        return p_v


class PersistentRBM(RBM):
    """
    RBM with Persistent Contrastive Divergence (PCD).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistent_chain = None
        self.lr = 0.01  # Default learning rate
    
    def pcd_step(self, v, k=1, lr=None):
        """
        Perform one step of persistent contrastive divergence.
        
        Parameters:
        -----------
        v: array-like
            Input data
        k: int
            Number of Gibbs sampling steps
        lr: float or None
            Learning rate (if None, use self.lr)
            
        Returns:
        --------
        float
            Reconstruction error
        """
        if lr is not None:
            self.lr = lr
            
        if self.persistent_chain is None:
            self.persistent_chain = np.random.binomial(1, 0.5, (v.shape[0], self.n_hidden))
        
        # Positive phase
        p_h_pos, _ = self.sample_hidden(v)
        
        # Negative phase - use persistent chain
        h_neg = self.persistent_chain
        for _ in range(k):
            p_v_neg, v_neg = self.sample_visible(h_neg)
            p_h_neg, h_neg = self.sample_hidden(v_neg)
        
        # Update persistent chain
        self.persistent_chain = h_neg
        
        # Compute gradients
        batch_size = v.shape[0]
        pos_associations = np.dot(v.T, p_h_pos)
        neg_associations = np.dot(p_v_neg.T, p_h_neg)
        
        # Update parameters
        grad_W = (pos_associations - neg_associations) / batch_size
        grad_a = np.mean(v - p_v_neg, axis=0)
        grad_b = np.mean(p_h_pos - p_h_neg, axis=0)
        
        self.W += self.lr * grad_W
        self.a += self.lr * grad_a
        self.b += self.lr * grad_b
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((v - p_v_neg) ** 2)
        return reconstruction_error


class TRBM(RBM):
    """
    RBM with temperature parameter.
    """
    def __init__(self, temp=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = temp
    
    def sample_hidden(self, v):
        activation = (np.dot(v, self.W) + self.b) / self.T
        p_h = self.sigmoid(activation)
        return p_h, (np.random.rand(*p_h.shape) < p_h).astype(float)
    
    def sample_visible(self, h):
        activation = (np.dot(h, self.W.T) + self.a) / self.T
        p_v = self.sigmoid(activation)
        return p_v, (np.random.rand(*p_v.shape) < p_v).astype(float)