import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Generator and Discriminator models
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128 * 8 * 8, input_dim=latent_dim),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Flatten(),
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

# Load CelebA dataset
def load_celeba(data_dir, img_size=(64, 64), batch_size=128):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode=None,  # No labels needed
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    # Normalize images to [-1, 1]
    dataset = dataset.map(lambda x: (x / 127.5) - 1.0)
    return dataset

# Train the GAN
def train_gan(generator, discriminator, gan, latent_dim, data_dir, epochs=10000, batch_size=128):
    data_generator = load_celeba(data_dir, img_size=(64, 64), batch_size=batch_size)

    for epoch in range(epochs):
        # Get a batch of real images
        real_images = next(iter(data_generator))

        # Train Discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

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
        axes[i].imshow(generated_images[i])
        axes[i].axis('off')
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("GAN Training with Streamlit")
    st.write("This app trains a GAN on the CelebA dataset and visualizes the generated faces.")

    # Parameters
    latent_dim = 100
    data_dir = "data/celeba"  # Path to CelebA images
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    compile_models(generator, discriminator, gan)

    # Streamlit UI
    epochs = st.slider("Number of Epochs", 100, 10000, 1000)
    batch_size = st.slider("Batch Size", 32, 256, 128)

    if st.button("Train GAN"):
        train_gan(generator, discriminator, gan, latent_dim, data_dir, epochs, batch_size)

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
    ax.imshow(smoothed_image[0])
    ax.axis('off')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
