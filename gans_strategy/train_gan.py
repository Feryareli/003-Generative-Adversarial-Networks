import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Construir Generador

def build_generator(latent_dim=300, seq_len=252):
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(256, activation='relu'),
         layers.Dense(256, activation='relu'),
        layers.Dense(seq_len, activation='linear')  # Salida con una característica por paso
    ])
    return model

# Discriminador (Critic)
def build_discriminator(seq_len=252):
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(seq_len,1)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')  # Salida lineal para WGAN
    ])
    return model

# Función para entrenar el GAN
def train_gan(generator, discriminator, x_train_norm, epochs=1000, batch_size=100, clip_value=1):
    gen_loss_history = []
    disc_loss_history = []

    # Optimizadores para el generador y el discriminador
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # Número de lotes
    num_batches = (len(x_train_norm) // batch_size) - 1


@tf.function
def train_step(real_data):
    noise = tf.random.normal([batch_size, generator.input_shape[-1]])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generar datos falsos
        generated_data = generator(noise, training=True)

        # Predicciones del discriminador
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)

        # Calcular pérdidas
        gen_loss = -tf.reduce_mean(fake_output)
        disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    # Calcular gradientes
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Aplicar gradientes
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Clipping de pesos para el discriminador (critic)
    for var in discriminator.trainable_variables:
        var.assign(tf.clip_by_value(var, -clip_value, clip_value))

    return gen_loss, disc_loss

    # Entrenar por varias épocas
    num_batches = (len(x_train_norm) // 252) -1
    for epoch in range(epochs):
        for batch in range(num_batches):
            batch = x_train_norm[batch*252:(batch+1)*252]
            gen_loss, disc_loss = train_step(batch, generator, discriminator)
            gen_loss_history.append(gen_loss)
            disc_loss_history.append(disc_loss)

    # Cada ciertas iteration, imprime las pérdidas
    print(f"Epoch {epoch}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")






