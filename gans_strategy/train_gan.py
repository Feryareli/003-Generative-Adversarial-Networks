import tensorflow as tf
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

# Funcion de paso
@tf.function
def train_step(real_data, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size=100, clip_value=1):
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