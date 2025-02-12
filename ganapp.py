import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Define the Generator and Discriminator models
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(784, activation='tanh'),
        layers.Reshape((28, 28))
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([
        generator,
        discriminator
    ])
    return model

# Compile the models
def compile_models(generator, discriminator, gan):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN
def train_gan(generator, discriminator, gan, latent_dim, epochs=10000, batch_size=128):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.0  # Normalize to [-1, 1]
    X_train = np.expand_dims(X_train, axis=-1)

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        if epoch % 100 == 0:
            st.write(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
            plot_generated_images(generator, latent_dim)

# Plot generated images
def plot_generated_images(generator, latent_dim, num_images=10):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    for i in range(num_images):
        axes[i].imshow(generated_images[i], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("GAN Training with Streamlit")
    st.write("This app trains a GAN on the MNIST dataset and visualizes the generated images.")

    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    compile_models(generator, discriminator, gan)

    epochs = st.slider("Number of Epochs", 100, 10000, 1000)
    batch_size = st.slider("Batch Size", 32, 256, 128)

    if st.button("Train GAN"):
        train_gan(generator, discriminator, gan, latent_dim, epochs, batch_size)

    st.write("Adjust the slider to see the generated images at different stages of training.")
    layer_slider = st.slider("Layer Smoothing", 0, 10, 5)
    st.write(f"Visualizing layer smoothing at level: {layer_slider}")

    # Visualize the effect of layer smoothing
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)
    generated_image = 0.5 * generated_image + 0.5  # Rescale to [0, 1]

    # Apply smoothing (this is a placeholder for actual smoothing logic)
    smoothed_image = generated_image * (layer_slider / 10.0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(smoothed_image[0], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

if __name__ == "__main__":
    main()