import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# Add joblib for parallelization
from joblib import Parallel, delayed

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
        'rbm': {'n_hidden': 20},
        'dbn': {'layer_sizes': [784, 20, 20]},
        'vae': {'encoder_dims': [784, 256, 128], 'latent_dim': 2, 'decoder_dims': [128, 256, 784]},
        'gan': {'g_input_dim': 2, 'g_hidden_dim': 6, 'g_output_dim': 784, 'g_depth': 1,
                'd_input_dim': 784, 'd_hidden_dim': 14, 'd_output_dim': 1, 'd_depth': 1}
    },
    'large': {
        'rbm': {'n_hidden': 100},
        'dbn': {'layer_sizes': [784, 82, 82, 82]},
        'vae': {'encoder_dims': [784, 512, 256], 'latent_dim': 10, 'decoder_dims': [256, 512, 784]},
        'gan': {'g_input_dim': 10, 'g_hidden_dim': 16, 'g_output_dim': 784, 'g_depth': 2,
                'd_input_dim': 784, 'd_hidden_dim': 64, 'd_output_dim': 1, 'd_depth': 2}
    },
    'xlarge': {
        'rbm': {'n_hidden': 500},
        'dbn': {'layer_sizes': [784, 256, 256, 256, 256]},
        'vae': {'encoder_dims': [784, 1024, 512], 'latent_dim': 64, 'decoder_dims': [512, 1024, 784]},
        'gan': {'g_input_dim': 32, 'g_hidden_dim': 64, 'g_output_dim': 784, 'g_depth': 3,
                'd_input_dim': 784, 'd_hidden_dim': 160, 'd_output_dim': 1, 'd_depth': 3}
    }
}

#===============================================================================
# VAE Implementation using Standard Keras
#===============================================================================

def build_vae(encoder_dims, latent_dim, decoder_dims):
    """Build VAE using a simpler, more reliable approach with subclassing."""
    class Sampling(layers.Layer):
        """Sampling layer for VAE."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    class VAE(keras.Model):
        """Variational Autoencoder model."""
        def __init__(self, encoder_dims, latent_dim, decoder_dims):
            super(VAE, self).__init__()
            self.latent_dim = latent_dim
            
            # Build encoder
            self.encoder_layers = []
            for dim in encoder_dims[1:]:
                self.encoder_layers.append(layers.Dense(dim, activation='relu'))
            
            self.z_mean = layers.Dense(latent_dim)
            self.z_log_var = layers.Dense(latent_dim)
            self.sampling = Sampling()
            
            # Build decoder
            self.decoder_layers = []
            for dim in decoder_dims[:-1]:
                self.decoder_layers.append(layers.Dense(dim, activation='relu'))
            self.decoder_output = layers.Dense(decoder_dims[-1], activation='sigmoid')
            
            # Optimizer
            self.optimizer = keras.optimizers.Adam(1e-3)
            
            # Track loss
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]
        
        def encode(self, x):
            for layer in self.encoder_layers:
                x = layer(x)
            z_mean = self.z_mean(x)
            z_log_var = self.z_log_var(x)
            z = self.sampling([z_mean, z_log_var])
            return z_mean, z_log_var, z
        
        def decode(self, z):
            x = z
            for layer in self.decoder_layers:
                x = layer(x)
            return self.decoder_output(x)
        
        def call(self, inputs):
            z_mean, z_log_var, z = self.encode(inputs)
            reconstructed = self.decode(z)
            return reconstructed
        
        def train_step(self, data):
            x = data
            
            with tf.GradientTape() as tape:
                # Forward pass
                z_mean, z_log_var, z = self.encode(x)
                reconstruction = self.decode(z)
                
                # Compute loss
                reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(x, reconstruction) * encoder_dims[0]
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                total_loss = reconstruction_loss + kl_loss
            
            # Compute gradients and update weights
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            # Update metrics
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
        
        def test_step(self, data):
            x = data
            
            # Forward pass
            z_mean, z_log_var, z = self.encode(x)
            reconstruction = self.decode(z)
            
            # Compute loss
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(x, reconstruction) * encoder_dims[0]
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = reconstruction_loss + kl_loss
            
            # Update metrics
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
        
        def generate_samples(self, n_samples):
            """Generate samples from the latent space."""
            z = tf.random.normal(shape=(n_samples, self.latent_dim))
            return self.decode(z).numpy()
    
    # Instantiate and return the VAE model
    vae = VAE(encoder_dims, latent_dim, decoder_dims)
    
    # Compile the model (not needed for custom training)
    vae.compile()
    
    return vae

#===============================================================================
# GAN Implementation using Standard Keras
#===============================================================================

def build_gan(g_input_dim, g_hidden_dim, g_output_dim, g_depth,
              d_input_dim, d_hidden_dim, d_output_dim, d_depth):
    """Build GAN models using standard Keras layers."""
    # Build generator using the functional API
    generator_input = keras.Input(shape=(g_input_dim,))
    x = layers.Dense(g_hidden_dim)(generator_input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.BatchNormalization()(x)
    
    # Hidden layers
    hidden_dims = [g_hidden_dim * (2**i) for i in range(1, g_depth)]
    for dim in hidden_dims:
        x = layers.Dense(dim)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.BatchNormalization()(x)
    
    # Output layer
    generator_output = layers.Dense(g_output_dim, activation='sigmoid')(x)
    
    # Create generator model
    generator = keras.Model(generator_input, generator_output, name="generator")
    
    # Build discriminator using the functional API
    discriminator_input = keras.Input(shape=(d_input_dim,))
    x = layers.Dense(d_hidden_dim)(discriminator_input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Hidden layers
    hidden_dims = [d_hidden_dim // (2**i) for i in range(1, d_depth)]
    for dim in hidden_dims:
        x = layers.Dense(dim)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dropout(0.3)(x)
    
    # Output layer
    discriminator_output = layers.Dense(d_output_dim, activation='sigmoid')(x)
    
    # Create discriminator model
    discriminator = keras.Model(discriminator_input, discriminator_output, name="discriminator")
    
    # Compile discriminator
    discriminator.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Build GAN using functional API
    gan_input = keras.Input(shape=(g_input_dim,))
    x = generator(gan_input)
    # Only when building the composite model, we set discriminator to not trainable
    discriminator.trainable = False
    gan_output = discriminator(x)
    gan = keras.Model(gan_input, gan_output, name="gan")
    
    # Compile GAN
    gan.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy'
    )
    
    # Create a class-like object with the models and generate_samples method
    class GANModel:
        def __init__(self, generator, discriminator, gan):
            self.generator = generator
            self.discriminator = discriminator
            self.gan = gan
            self.g_input_dim = g_input_dim
        
        def generate_samples(self, n_samples):
            noise = np.random.normal(0, 1, size=(n_samples, self.g_input_dim))
            return self.generator.predict(noise, verbose=0)
    
    return GANModel(generator, discriminator, gan)

def train_gan(X_train, X_val=None, config=None, nb_epochs=100, batch_size=100, model_name="gan_small"):
    """Train a GAN model with given configuration using standard Keras."""
    print(f"Training GAN with generator: {config['g_input_dim']}->{config['g_hidden_dim']}->{config['g_output_dim']} (depth: {config['g_depth']}), "
          f"discriminator: {config['d_input_dim']}->{config['d_hidden_dim']}->{config['d_output_dim']} (depth: {config['d_depth']})...")
    
    # Build GAN
    gan_model = build_gan(
        g_input_dim=config['g_input_dim'],
        g_hidden_dim=config['g_hidden_dim'],
        g_output_dim=config['g_output_dim'],
        g_depth=config['g_depth'],
        d_input_dim=config['d_input_dim'],
        d_hidden_dim=config['d_hidden_dim'],
        d_output_dim=config['d_output_dim'],
        d_depth=config['d_depth']
    )
    
    # Train the GAN model
    d_losses = []
    g_losses = []
    
    for epoch in range(nb_epochs):
        start_time = time()
        
        d_epoch_loss = 0
        g_epoch_loss = 0
        num_batches = 0
        
        # Create dataset and shuffle
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        
        # Train in batches
        for i in range(0, len(X_train), batch_size):
            # Get batch
            X_batch = X_shuffled[i:i+batch_size]
            batch_size_actual = len(X_batch)
            
            # Generate fake samples
            noise = np.random.normal(0, 1, size=(batch_size_actual, config['g_input_dim']))
            generated_images = gan_model.generator.predict(noise, verbose=0)
            
            # Labels for real and fake samples
            real_labels = np.ones((batch_size_actual, 1))
            fake_labels = np.zeros((batch_size_actual, 1))
            
            # Train discriminator
            gan_model.discriminator.trainable = True
            d_loss_real = gan_model.discriminator.train_on_batch(X_batch, real_labels)[0]
            d_loss_fake = gan_model.discriminator.train_on_batch(generated_images, fake_labels)[0]
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            # Train generator
            gan_model.discriminator.trainable = False
            noise = np.random.normal(0, 1, size=(batch_size_actual*2, config['g_input_dim']))
            misleading_labels = np.ones((batch_size_actual*2, 1))
            g_loss = gan_model.gan.train_on_batch(noise, misleading_labels)
            
            # Update epoch losses
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
            num_batches += 1
        
        # Calculate average losses for the epoch
        d_epoch_loss /= num_batches
        g_epoch_loss /= num_batches
        d_losses.append(d_epoch_loss)
        g_losses.append(g_epoch_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{nb_epochs} - {time() - start_time:.2f}s - d_loss: {d_epoch_loss:.4f} - g_loss: {g_epoch_loss:.4f}")
        
        # Validate if validation data is provided
        if X_val is not None and (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            val_samples = min(1000, len(X_val))
            idx = np.random.choice(len(X_val), val_samples, replace=False)
            X_val_sample = X_val[idx]
            
            # Make sure discriminator is trainable for evaluation
            gan_model.discriminator.trainable = True
            
            # Generate fake samples
            noise = np.random.normal(0, 1, size=(val_samples, config['g_input_dim']))
            fake_samples = gan_model.generator.predict(noise, verbose=0)
            
            # Evaluate discriminator
            d_loss_real = gan_model.discriminator.evaluate(X_val_sample, np.ones((val_samples, 1)), verbose=0)[0]
            d_loss_fake = gan_model.discriminator.evaluate(fake_samples, np.zeros((val_samples, 1)), verbose=0)[0]
            d_val_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            # Print validation loss
            print(f"  Validation - d_loss: {d_val_loss:.4f}")
    
    # Save models with proper file extension (.keras or .h5)
    gan_model.generator.save(f"results/models/comparison/{model_name}_generator.keras")
    gan_model.discriminator.save(f"results/models/comparison/{model_name}_discriminator.keras")
    
    # Save some samples as well
    samples = gan_model.generate_samples(100)
    samples_path = f"results/models/comparison/{model_name}_samples.npy"
    np.save(samples_path, samples)
    
    return gan_model

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
    
    # Generate and save samples for consistency with other models
    samples = rbm.generate_samples(100, gibbs_steps=500)
    samples_path = f"results/models/comparison/{model_name}_samples.npy"
    np.save(samples_path, samples)
    
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
    """Train a VAE model with given configuration using standard Keras."""
    print(f"Training VAE with encoder: {config['encoder_dims']}, latent dim: {config['latent_dim']}, decoder: {config['decoder_dims']}...")
    
    # Build VAE
    vae = build_vae(
        encoder_dims=config['encoder_dims'],
        latent_dim=config['latent_dim'],
        decoder_dims=config['decoder_dims']
    )
    
    # Train the model
    history = vae.fit(
        X_train,  # For custom training, we only need the input 
        epochs=nb_epochs,
        batch_size=batch_size,
        validation_data=X_val if X_val is not None else None,
        verbose=1
    )
    
    # Instead of saving weights directly, save generated samples
    print("Generating and saving samples from VAE...")
    samples = vae.generate_samples(100)  # Generate more samples than needed
    
    # Save samples to a numpy file
    samples_path = f"results/models/comparison/{model_name}_samples.npy"
    np.save(samples_path, samples)
    
    print(f"VAE samples saved to {samples_path}")
    
    return vae

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
    # Model labels and size labels
    model_labels = {'rbm': 'RBM', 'dbn': 'DBN', 'vae': 'VAE', 'gan': 'GAN'}
    size_labels = {'small': 'Small', 'large': 'Large', 'xlarge': 'XLarge'}
    
    # Generate samples for each model and size
    for model_type in ['rbm', 'dbn', 'vae', 'gan']:
        for size in ['small', 'large', 'xlarge']:
            print(f"Generating samples for {model_type.upper()} - {size.upper()}")
            
            # Create a figure for this model and configuration
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            axes = axes.flatten()
            
            # Get model if available
            model = models.get(model_type, {}).get(size, None)
            
            # Generate or load samples
            if model_type in ['rbm', 'dbn'] and model is not None:
                # Generate samples for RBM and DBN from model - standard 500 Gibbs steps
                samples = model.generate_samples(n_samples, gibbs_steps=500)
                samples = samples[:25]  # Take first 25 samples
            elif model_type in ['vae', 'gan']:
                if model is not None:
                    # Generate samples from proxy models
                    samples = model.generate_samples(25)
                else:
                    # Try to load samples from saved files
                    samples_path = f"results/models/comparison/{model_type}_{size}_samples.npy"
                    try:
                        samples = np.load(samples_path)
                        samples = samples[:25]  # Take first 25 samples
                        print(f"Loaded samples from {samples_path}")
                    except:
                        print(f"WARNING: Could not load samples for {model_type}_{size}")
                        # Create empty samples as fallback
                        samples = np.zeros((25, 784))
            else:
                print(f"WARNING: No model found for {model_type}_{size}")
                # Create empty samples as fallback
                samples = np.zeros((25, 784))
            
            # Display 5x5 grid of samples
            for k in range(min(25, len(samples))):
                sample = samples[k].reshape(28, 28)
                axes[k].imshow(sample, cmap='gray')
                axes[k].axis('off')
            
            # Set title for the figure
            plt.suptitle(f'{model_labels[model_type]} - {size_labels[size]}', fontsize=16)
            
            # Save individual figure
            plot_path = f"results/plots/comparison/{model_type}_{size}_{dataset}.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot to {plot_path}")
    
    # Print completion message
    print(f"Model evaluation completed. Generated samples saved to results/plots/comparison/")

#===============================================================================
# Proxy classes for pickling models
#===============================================================================

# Define proxy classes at module level so they can be pickled
class VaeProxy:
    """Proxy class for VAE to make it picklable."""
    def __init__(self, samples):
        self.samples = samples
        
    def generate_samples(self, n_samples):
        # Return a subset of the pre-generated samples
        return self.samples[:n_samples]


class GanProxy:
    """Proxy class for GAN to make it picklable."""
    def __init__(self, samples):
        self.samples = samples
        
    def generate_samples(self, n_samples):
        return self.samples[:n_samples]

#===============================================================================
# Parallel Training Function
#===============================================================================

def train_model_for_size(model_type, size, X_train, X_val, config, nb_epochs, batch_size, learning_rate):
    """Wrapper function to train a specific model type with given size configuration."""
    print(f"Starting training {model_type.upper()} - {size.upper()}")
    
    model_name = f"{model_type}_{size}"
    
    if model_type == 'rbm':
        model = train_rbm(
            X_train, X_val, 
            config=config, 
            nb_epochs=nb_epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            model_name=model_name
        )
        # For RBM and DBN, we return the actual model since they're picklable
        return model_type, size, model
        
    elif model_type == 'dbn':
        model = train_dbn(
            X_train, X_val, 
            config=config, 
            nb_epochs=nb_epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            model_name=model_name
        )
        # For RBM and DBN, we return the actual model since they're picklable
        return model_type, size, model
        
    elif model_type == 'vae':
        model = train_vae(
            X_train, X_val, 
            config=config, 
            nb_epochs=nb_epochs, 
            batch_size=batch_size,
            model_name=model_name
        )
        # For VAE, don't return the model (it's not picklable)
        # Instead, generate samples and return a proxy object with the generate_samples method
        samples = model.generate_samples(100)
        
        # Use the module-level VaeProxy class instead of defining it here
        proxy = VaeProxy(samples)
        
        # Verify that samples can be properly accessed
        test_sample = proxy.generate_samples(1)
        print(f"VAE proxy sample shape: {test_sample.shape}")
        
        return model_type, size, proxy
        
    elif model_type == 'gan':
        model = train_gan(
            X_train, X_val, 
            config=config, 
            nb_epochs=nb_epochs, 
            batch_size=batch_size,
            model_name=model_name
        )
        # For GAN, similarly create a proxy with pre-generated samples
        samples = model.generate_samples(100)
        
        # Use the module-level GanProxy class instead of defining it here
        proxy = GanProxy(samples)
        
        # Verify that samples can be properly accessed
        test_sample = proxy.generate_samples(1)
        print(f"GAN proxy sample shape: {test_sample.shape}")
        
        return model_type, size, proxy
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
    
    # Create a list of training tasks
    training_tasks = []
    for size in ['small', 'large', 'xlarge']:
        for model_type in ['rbm', 'dbn', 'vae', 'gan']:
            config = MODEL_CONFIGS[size][model_type]
            training_tasks.append(
                delayed(train_model_for_size)(
                    model_type, size, X_train, X_val, 
                    config, nb_epochs, batch_size, learning_rate
                )
            )
    
    # Run training tasks in parallel - limit to 4 processes to avoid GPU memory issues
    print(f"\n=== Starting parallel training of {len(training_tasks)} models ===\n")
    results = Parallel(n_jobs=-1, verbose=10)(training_tasks)
    
    # Organize results into the trained_models dictionary
    for model_type, size, model in results:
        trained_models[model_type][size] = model
    
    # Evaluate models
    print("\n=== Evaluating models ===\n")
    # Ensure all trained models are included
    print(f"Models ready for evaluation: {list(trained_models.keys())}")
    for model_type in trained_models:
        print(f"  {model_type}: {list(trained_models[model_type].keys())}")
    
    evaluate_models(trained_models, n_samples=25, dataset="mnist")
    
    # Also save the trained_models dictionary for potential later use
    os.makedirs("results/models/comparison", exist_ok=True)
    with open("results/models/comparison/trained_model_proxies.pkl", "wb") as f:
        pickle.dump(trained_models, f)
    
    print("\nAll model training and evaluation completed!")

if __name__ == "__main__":
    main()
