import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split

# Import from our own modules
from models import RBM, DBN
from utils import (load_mnist, 
                  load_binary_alphadigits,
                  create_data_splits, 
                  display_binary_images)

# Create directories for saving results if they don't exist
os.makedirs("results/plots/comparison", exist_ok=True)
os.makedirs("results/models/comparison", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#===============================================================================
# Model Configuration
#===============================================================================

# Configuration dictionary as per table
MODEL_CONFIGS = {
    'small': {
        'rbm': {'n_hidden': 20, 'params': '~16484'},
        'dbn': {'layer_sizes': [784, 20, 20], 'params': '~16924'},
        'vae': {'encoder_dims': [784, 256, 128], 'latent_dim': 2, 'decoder_dims': [128, 256, 784], 'params': '~16768'},
        'gan': {'g_input_dim': 2, 'g_hidden_dim': 6, 'g_output_dim': 784, 'g_depth': 1,
                'd_input_dim': 784, 'd_hidden_dim': 14, 'd_output_dim': 1, 'd_depth': 1, 'params': '~16511'}
    },
    'large': {
        'rbm': {'n_hidden': 100, 'params': '~79284'},
        'dbn': {'layer_sizes': [784, 82, 82, 82], 'params': '~78930'},
        'vae': {'encoder_dims': [784, 512, 256], 'latent_dim': 10, 'decoder_dims': [256, 512, 784], 'params': '~79236'},
        'gan': {'g_input_dim': 10, 'g_hidden_dim': 16, 'g_output_dim': 784, 'g_depth': 2,
                'd_input_dim': 784, 'd_hidden_dim': 64, 'd_output_dim': 1, 'd_depth': 2, 'params': '~78945'}
    },
    'xlarge': {
        'rbm': {'n_hidden': 500, 'params': '~393284'},
        'dbn': {'layer_sizes': [784, 256, 256, 256, 256], 'params': '~399888'},
        'vae': {'encoder_dims': [784, 1024, 512], 'latent_dim': 64, 'decoder_dims': [512, 1024, 784], 'params': '~390944'},
        'gan': {'g_input_dim': 32, 'g_hidden_dim': 64, 'g_output_dim': 784, 'g_depth': 3,
                'd_input_dim': 784, 'd_hidden_dim': 160, 'd_output_dim': 1, 'd_depth': 3, 'params': '~386705'}
    }
}

#===============================================================================
# VAE Implementation
#===============================================================================

class VAE(Model):
    def __init__(self, encoder_dims, latent_dim, decoder_dims):
        super(VAE, self).__init__()
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.decoder_dims = decoder_dims
        
        # Build the encoder
        self.encoder_layers = []
        for dim in encoder_dims[1:]:
            self.encoder_layers.append(layers.Dense(dim, activation='relu'))
        
        # Mean and variance layers
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)
        
        # Build the decoder
        self.decoder_layers = []
        for dim in decoder_dims[:-1]:
            self.decoder_layers.append(layers.Dense(dim, activation='relu'))
        self.decoder_output = layers.Dense(decoder_dims[-1], activation='sigmoid')
        
        # Compile the model
        self.compile(optimizer=optimizers.Adam(1e-3))
    
    def encode(self, x):
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Get mean and log variance
        mean = self.mean(x)
        log_var = self.log_var(x)
        
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
    def decode(self, z):
        # Pass through decoder layers
        x = z
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final output layer
        return self.decoder_output(x)
    
    def call(self, x):
        # Encode
        mean, log_var = self.encode(x)
        
        # Sample latent vector
        z = self.reparameterize(mean, log_var)
        
        # Decode
        reconstructed = self.decode(z)
        
        # Add loss
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        self.add_loss(tf.reduce_mean(kl_loss))
        
        return reconstructed
    
    def generate_samples(self, n_samples):
        # Sample from latent space
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        
        # Decode
        samples = self.decode(z)
        
        return samples.numpy()
    
    def fit(self, x_train, x_val=None, epochs=100, batch_size=128, verbose=1):
        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        val_dataset = None
        if x_val is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
            val_dataset = val_dataset.batch(batch_size)
        
        # Custom training loop
        self.losses = []
        self.val_losses = [] if x_val is not None else None
        
        for epoch in range(epochs):
            start_time = time()
            
            # Training
            epoch_loss = 0
            num_batches = 0
            
            for x_batch in train_dataset:
                with tf.GradientTape() as tape:
                    # Forward pass
                    reconstructed = self(x_batch)
                    
                    # Reconstruction loss
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            keras.losses.binary_crossentropy(x_batch, reconstructed),
                            axis=(1)
                        )
                    )
                    
                    # Total loss
                    total_loss = reconstruction_loss + sum(self.losses) / x_batch.shape[0]
                
                # Backpropagation
                grads = tape.gradient(total_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                
                epoch_loss += total_loss.numpy()
                num_batches += 1
            
            epoch_loss /= num_batches
            self.losses.append(epoch_loss)
            
            # Validation
            if val_dataset:
                val_loss = 0
                val_batches = 0
                
                for x_val_batch in val_dataset:
                    # Forward pass
                    reconstructed = self(x_val_batch)
                    
                    # Reconstruction loss
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            keras.losses.binary_crossentropy(x_val_batch, reconstructed),
                            axis=(1)
                        )
                    )
                    
                    # Total loss
                    total_loss = reconstruction_loss + sum(self.losses) / x_val_batch.shape[0]
                    
                    val_loss += total_loss.numpy()
                    val_batches += 1
                
                val_loss /= val_batches
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time() - start_time:.2f}s - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time() - start_time:.2f}s - loss: {epoch_loss:.4f}")
        
        return self.losses

#===============================================================================
# GAN Implementation
#===============================================================================

class GAN:
    def __init__(self, g_input_dim, g_hidden_dim, g_output_dim, g_depth, 
                 d_input_dim, d_hidden_dim, d_output_dim, d_depth):
        self.g_input_dim = g_input_dim
        self.g_hidden_dim = g_hidden_dim
        self.g_output_dim = g_output_dim
        self.g_depth = g_depth
        
        self.d_input_dim = d_input_dim
        self.d_hidden_dim = d_hidden_dim
        self.d_output_dim = d_output_dim
        self.d_depth = d_depth
        
        # Build generator
        self.generator = self._build_generator()
        
        # Build discriminator
        self.discriminator = self._build_discriminator()
        
        # Build GAN
        self.gan = self._build_gan()
        
        # Compile models
        self.discriminator.compile(
            optimizer=optimizers.Adam(1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.gan.compile(
            optimizer=optimizers.Adam(1e-4),
            loss='binary_crossentropy'
        )
    
    def _build_generator(self):
        model = keras.Sequential()
        
        # First layer
        model.add(layers.Dense(self.g_hidden_dim, input_dim=self.g_input_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        
        # Hidden layers
        hidden_dims = [self.g_hidden_dim * (2**i) for i in range(1, self.g_depth)]
        for dim in hidden_dims:
            model.add(layers.Dense(dim))
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.BatchNormalization())
        
        # Output layer
        model.add(layers.Dense(self.g_output_dim, activation='sigmoid'))
        
        return model
    
    def _build_discriminator(self):
        model = keras.Sequential()
        
        # First layer
        model.add(layers.Dense(self.d_hidden_dim, input_dim=self.d_input_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        # Hidden layers
        hidden_dims = [self.d_hidden_dim // (2**i) for i in range(1, self.d_depth)]
        for dim in hidden_dims:
            model.add(layers.Dense(dim))
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(self.d_output_dim, activation='sigmoid'))
        
        return model
    
    def _build_gan(self):
        # Make discriminator not trainable for the GAN model
        self.discriminator.trainable = False
        
        # Connect generator and discriminator
        model = keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        
        return model
    
    def generate_samples(self, n_samples):
        # Generate random noise
        noise = np.random.normal(0, 1, size=(n_samples, self.g_input_dim))
        
        # Generate samples
        samples = self.generator.predict(noise)
        
        return samples
    
    def fit(self, x_train, x_val=None, epochs=100, batch_size=128, verbose=1):
        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        # For tracking progress
        self.losses = {'d_loss': [], 'g_loss': []}
        self.val_losses = {'d_loss': [], 'g_loss': []} if x_val is not None else None
        
        for epoch in range(epochs):
            start_time = time()
            
            # Training
            d_epoch_loss = 0
            g_epoch_loss = 0
            num_batches = 0
            
            for x_batch in train_dataset:
                batch_size = x_batch.shape[0]
                
                # Generate noise
                noise = np.random.normal(0, 1, size=(batch_size, self.g_input_dim))
                
                # Generate fake samples
                fake_samples = self.generator.predict(noise)
                
                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(x_batch, np.ones((batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                
                # Train generator
                noise = np.random.normal(0, 1, size=(batch_size * 2, self.g_input_dim))
                g_loss = self.gan.train_on_batch(noise, np.ones((batch_size * 2, 1)))
                
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss
                num_batches += 1
            
            d_epoch_loss /= num_batches
            g_epoch_loss /= num_batches
            
            self.losses['d_loss'].append(d_epoch_loss)
            self.losses['g_loss'].append(g_epoch_loss)
            
            # Validation
            if x_val is not None:
                val_samples = min(1000, x_val.shape[0])
                idx = np.random.choice(x_val.shape[0], val_samples, replace=False)
                x_val_sample = x_val[idx]
                
                # Generate fake samples
                noise = np.random.normal(0, 1, size=(val_samples, self.g_input_dim))
                fake_samples = self.generator.predict(noise)
                
                # Evaluate discriminator
                d_loss_real = self.discriminator.evaluate(x_val_sample, np.ones((val_samples, 1)), verbose=0)
                d_loss_fake = self.discriminator.evaluate(fake_samples, np.zeros((val_samples, 1)), verbose=0)
                d_val_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                
                # Evaluate generator
                noise = np.random.normal(0, 1, size=(val_samples * 2, self.g_input_dim))
                g_val_loss = self.gan.evaluate(noise, np.ones((val_samples * 2, 1)), verbose=0)
                
                self.val_losses['d_loss'].append(d_val_loss)
                self.val_losses['g_loss'].append(g_val_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time() - start_time:.2f}s - d_loss: {d_epoch_loss:.4f} - g_loss: {g_epoch_loss:.4f} - val_d_loss: {d_val_loss:.4f} - val_g_loss: {g_val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time() - start_time:.2f}s - d_loss: {d_epoch_loss:.4f} - g_loss: {g_epoch_loss:.4f}")
        
        return self.losses

#===============================================================================
# Model Training Functions
#===============================================================================

def train_rbm(X_train, X_val=None, config=None, nb_epochs=100, batch_size=100, learning_rate=0.01, model_name="rbm_small"):
    """Train an RBM model with given configuration."""
    print(f"Training RBM with {config['n_hidden']} hidden units...")
    
    # Initialize RBM
    rbm = RBM(n_visible=X_train.shape[1], n_hidden=config['n_hidden'])
    
    # Train the model
    rbm.fit(X_train,
            validation_data=X_val,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            verbose=True)
    
    # Save the model
    model_path = f"results/models/comparison/{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(rbm, f)
    
    return rbm

def train_dbn(X_train, X_val=None, config=None, nb_epochs=100, batch_size=100, learning_rate=0.01, model_name="dbn_small"):
    """Train a DBN model with given configuration."""
    print(f"Training DBN with layer sizes: {config['layer_sizes']}...")
    
    # Initialize DBN
    dbn = DBN(layer_sizes=config['layer_sizes'])
    
    # Train the model
    dbn.fit(X_train,
            validation_data=X_val,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            verbose=True)
    
    # Save the model
    model_path = f"results/models/comparison/{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(dbn, f)
    
    return dbn

def train_vae(X_train, X_val=None, config=None, nb_epochs=100, batch_size=100, model_name="vae_small"):
    """Train a VAE model with given configuration."""
    print(f"Training VAE with encoder: {config['encoder_dims']}, latent dim: {config['latent_dim']}, decoder: {config['decoder_dims']}...")
    
    # Initialize VAE
    vae = VAE(encoder_dims=config['encoder_dims'], 
              latent_dim=config['latent_dim'], 
              decoder_dims=config['decoder_dims'])
    
    # Train the model
    vae.fit(X_train, X_val, epochs=nb_epochs, batch_size=batch_size, verbose=1)
    
    # Save the model
    model_path = f"results/models/comparison/{model_name}.h5"
    vae.save_weights(model_path)
    
    return vae

def train_gan(X_train, X_val=None, config=None, nb_epochs=100, batch_size=100, model_name="gan_small"):
    """Train a GAN model with given configuration."""
    print(f"Training GAN with generator: {config['g_input_dim']}->{config['g_hidden_dim']}->{config['g_output_dim']} (depth: {config['g_depth']}), "
          f"discriminator: {config['d_input_dim']}->{config['d_hidden_dim']}->{config['d_output_dim']} (depth: {config['d_depth']})...")
    
    # Initialize GAN
    gan = GAN(g_input_dim=config['g_input_dim'],
              g_hidden_dim=config['g_hidden_dim'],
              g_output_dim=config['g_output_dim'],
              g_depth=config['g_depth'],
              d_input_dim=config['d_input_dim'],
              d_hidden_dim=config['d_hidden_dim'],
              d_output_dim=config['d_output_dim'],
              d_depth=config['d_depth'])
    
    # Train the model
    gan.fit(X_train, X_val, epochs=nb_epochs, batch_size=batch_size, verbose=1)
    
    # Save the models
    gan.generator.save(f"results/models/comparison/{model_name}_generator.h5")
    gan.discriminator.save(f"results/models/comparison/{model_name}_discriminator.h5")
    
    return gan

#===============================================================================
# Evaluation Functions
#===============================================================================

def evaluate_models(models, n_samples=25, dataset="mnist"):
    """
    Evaluate generative performance of models by generating samples.
    
    Parameters:
    -----------
    models: dict
        Dictionary containing trained models
    n_samples: int
        Number of samples to generate from each model
    dataset: str
        Dataset name for figure title
    """
    # Number of models and sizes
    n_models = len(models)
    n_sizes = len(models['rbm'])
    
    # Create a figure
    fig, axes = plt.subplots(n_models, n_sizes, figsize=(5*n_sizes, 5*n_models))
    
    # Model labels and size labels
    model_labels = {'rbm': 'RBM', 'dbn': 'DBN', 'vae': 'VAE', 'gan': 'GAN'}
    size_labels = {'small': 'Small', 'large': 'Large', 'xlarge': 'XLarge'}
    
    # Generate samples for each model and size
    for i, (model_name, model_group) in enumerate(models.items()):
        for j, (size, model) in enumerate(model_group.items()):
            # Generate samples
            if model_name in ['rbm', 'dbn']:
                samples = model.generate_samples(n_samples, gibbs_steps=200)
                samples = samples[:25]  # Take first 25 samples
            else:  # VAE or GAN
                samples = model.generate_samples(25)
            
            # Display 5x5 grid of samples
            for k in range(min(25, len(samples))):
                row, col = k // 5, k % 5
                sample = samples[k].reshape(28, 28)
                
                # Calculate subplot index
                ax = axes[i][j] if n_sizes > 1 else axes[i]
                ax.imshow(sample, cmap='gray')
                ax.axis('off')
                ax.set_title(f'{model_labels[model_name]} - {size_labels[size]}')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"results/plots/comparison/model_comparison_{dataset}.png")
    plt.close()

    # Print completion message
    print(f"Model evaluation completed. Generated samples saved to results/plots/comparison/model_comparison_{dataset}.png")

#===============================================================================
# Main Function
#===============================================================================

def main():
    """Main function to run the generative model comparison."""
    # Loading MNIST dataset
    print("Loading MNIST dataset...")
    X_train_full, _, X_test, _ = load_mnist(binarize_threshold=0.5, normalize=True, use_cache=True)
    
    # Limit to 10,000 samples to reduce computational time
    if len(X_train_full) > 10000:
        print(f"Limiting training data to 10,000 samples (from {len(X_train_full)})")
        indices = np.random.choice(len(X_train_full), 10000, replace=False)
        X_train_full = X_train_full[indices]
    
    # Split into train and validation sets
    X_train, X_val = train_test_split(X_train_full, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Dictionary to store trained models
    trained_models = {
        'rbm': {}, 
        'dbn': {}, 
        'vae': {}, 
        'gan': {}
    }
    
    # Training parameters
    nb_epochs = 50  # Reduced for faster execution
    batch_size = 100
    learning_rate = 0.01
    
    # Train models for each size configuration
    for size in ['small', 'large', 'xlarge']:
        print(f"\n=== Training {size.upper()} models ===\n")
        
        # Train RBM
        rbm = train_rbm(
            X_train, X_val, 
            config=MODEL_CONFIGS[size]['rbm'], 
            nb_epochs=nb_epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            model_name=f"rbm_{size}"
        )
        trained_models['rbm'][size] = rbm
        
        # Train DBN
        dbn = train_dbn(
            X_train, X_val, 
            config=MODEL_CONFIGS[size]['dbn'], 
            nb_epochs=nb_epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            model_name=f"dbn_{size}"
        )
        trained_models['dbn'][size] = dbn
        
        # Train VAE
        vae = train_vae(
            X_train, X_val, 
            config=MODEL_CONFIGS[size]['vae'], 
            nb_epochs=nb_epochs, 
            batch_size=batch_size,
            model_name=f"vae_{size}"
        )
        trained_models['vae'][size] = vae
        
        # Train GAN
        gan = train_gan(
            X_train, X_val, 
            config=MODEL_CONFIGS[size]['gan'], 
            nb_epochs=nb_epochs, 
            batch_size=batch_size,
            model_name=f"gan_{size}"
        )
        trained_models['gan'][size] = gan
    
    # Evaluate models
    print("\n=== Evaluating models ===\n")
    evaluate_models(trained_models, n_samples=25, dataset="mnist")
    
    print("\nAll model training and evaluation completed!")

if __name__ == "__main__":
    main()
