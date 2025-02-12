import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define the Generator and Discriminator models
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128 * 8 * 8, input_dim=latent_dim, name="generator_dense_1"),
        layers.Reshape((8, 8, 128), name="generator_reshape_1"),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', name="generator_conv2d_transpose_1"),
        layers.LeakyReLU(alpha=0.2, name="generator_leaky_relu_1"),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', name="generator_conv2d_transpose_2"),
        layers.LeakyReLU(alpha=0.2, name="generator_leaky_relu_2"),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh', name="generator_conv2d_transpose_3")
    ], name="generator")
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3), name="discriminator_conv2d_1"),
        layers.LeakyReLU(alpha=0.2, name="discriminator_leaky_relu_1"),
        layers.Dropout(0.3),  # Add dropout for regularization
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same', name="discriminator_conv2d_2"),
        layers.LeakyReLU(alpha=0.2, name="discriminator_leaky_relu_2"),
        layers.Dropout(0.3),  # Add dropout for regularization
        layers.Flatten(name="discriminator_flatten_1"),
        layers.Dense(1, activation='sigmoid', name="discriminator_dense_1")
    ], name="discriminator")
    return model

# Define the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([
        generator,
        discriminator
    ], name="gan")
    return model

# Compile the models
def compile_models(generator, discriminator, gan):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer='adam', loss='binary_crossentropy')

# Custom function to load and augment images
def load_custom_images(data_dir, img_size=(64, 64)):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    images = []
    for path in image_paths:
        img = Image.open(path).resize(img_size)
        img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
        images.append(img)
        # Augment the image by flipping it horizontally
        images.append(np.fliplr(img))  # Add flipped version
    return np.array(images)

# Train the GAN
def train_gan(generator, discriminator, gan, latent_dim, data_dir, epochs=10000, batch_size=128):
    try:
        images = load_custom_images(data_dir, img_size=(64, 64))
        dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)
    except ValueError as e:
        st.error(str(e))  # Display error message in Streamlit
        return

    for epoch in range(epochs):
        for real_images in dataset:
            # Dynamically adjust the batch size based on the actual number of images
            current_batch_size = real_images.shape[0]

            # Train Discriminator
            noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
            fake_images = generator.predict(noise)

            real_labels = np.ones((current_batch_size, 1))
            fake_labels = np.zeros((current_batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
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
    st.write("This app trains a GAN on a small dataset and visualizes the generated faces.")

    # Parameters
    latent_dim = 100
    data_dir = "data/custom_images"  # Path to your custom images
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    compile_models(generator, discriminator, gan)

    # Streamlit UI
    epochs = st.slider("Number of E
