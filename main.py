import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm
from utils.utils import download_data, calculate_returns, normalize_returns, denormalize_returns
from gans_strategy.train_gan import build_generator, build_discriminator, train_step
from gans_strategy.backtest import passive_strategy
from gans_strategy.strategy import rsi_trading_strategy

# Configuración de los parámetros del proyecto
ticker = 'AAPL'
start_date = '2014-11-01'
end_date = '2024-11-01'

# Obtener x_train_norm
data = download_data(ticker, start_date, end_date)
returns = calculate_returns(data)
returns_norm = normalize_returns(returns)
x_train_norm = returns_norm.values  # Convertimos a valores numpy para el GAN

latent_dim = 300
num_scenarios = 100
seq_len = 252
data_points = len(data)

# Crear (instanciar) el modelo generador
generator = build_generator(latent_dim=latent_dim, seq_len=seq_len)

# Crear (instanciar) el modelo discriminador
discriminator = build_discriminator(seq_len=seq_len)

# Ver el resumen del generador (opcional)
generator.summary()

# Entrenar el GAN
gen_loss_history = []
disc_loss_history = []

generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
discrimitator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

epochs = 1_000
num_batches = (len(x_train_norm) // 252) - 1
for epoch in range(epochs):
    for batch in range(num_batches):
        batch = x_train_norm[batch*252:(batch+1)*252]
        gen_loss, disc_loss = train_step(batch, generator, discriminator, generator_optimizer, discrimitator_optimizer)
        gen_loss_history.append(gen_loss)
        disc_loss_history.append(disc_loss)

    print(f"Epoch {epoch}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")

# Generar escenarios
scenarios = []

for _ in range(num_scenarios):
    # Seleccionar un punto de inicio aleatorio del historial de precios
    start_index = np.random.choice(data_points - seq_len)  # Asegura que haya datos suficientes para la longitud de la secuencia
    initial_seed = returns_norm.values[start_index:start_index + seq_len]  # Extraer la semilla de longitud seq_len

    # Generar el vector de ruido para el generador
    noise = tf.random.normal([1, latent_dim])

    # Crear el escenario con el generador usando la semilla inicial
    generated_scenario = generator(noise, training=False).numpy().flatten()

    # Agregar el escenario generado a la lista de escenarios
    scenarios.append(generated_scenario)

# Convertir escenarios en un arreglo de NumPy para facilitar el backtesting
scenarios_array = np.array(scenarios)  # Shape: (100, 252)

# Graficar algunos de los escenarios sinteticos
plt.figure(figsize=(12, 6))
for i in range(min(num_scenarios, 100)):  # Graficamos solo 100 escenarios para mayor claridad
    plt.plot(scenarios_array[i], alpha=0.7, label=f"Escenario {i+1}")

plt.title("Rendimientos Sintéticos Generados por el GAN")
plt.xlabel("Días")
plt.ylabel("Rendimiento")
plt.show()

mean_return = returns.mean().vaules[0]
std_dev_return = returns.std().values[0]

# Generar fechas y datos ficticios para la serie original
dates = pd.date_range(start='2024-01-01', periods=seq_len, freq='B')  # Fechas de días hábiles para la simulación