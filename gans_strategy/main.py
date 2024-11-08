import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm
from utils.utils import download_data, calculate_returns, normalize_returns
from gans_strategy.train_gan import train_gan, generator
from gans_strategy.backtest import backtest_strategy, simple_trading_strategy, passive_strategy
from gans_strategy.strategy import generate_signals

# Configuración de los parámetros del proyecto
ticker = 'AAPL'
start_date = '2014-11-01'
end_date = '2024-11-01'
latent_dim = 300
num_scenarios = 100
seq_len = 252

# 1. Descarga y preprocesamiento de datos
data = download_data(ticker, start_date, end_date)
returns = calculate_returns(data)
returns_norm = normalize_returns(returns)
x_train_norm = returns_norm.values  # Convertimos a valores numpy para el GAN

# 2. Generación de escenarios
scenarios = []
data_points = len(data)  # Total de puntos de datos disponibles en el historial

for _ in range(num_scenarios):
    # Seleccionar un punto de inicio aleatorio del historial de precios
    start_index = np.random.choice(
        data_points - seq_len)  # Asegura que haya datos suficientes para la longitud de la secuencia
    initial_seed = returns_norm.values[start_index:start_index + seq_len]  # Extraer la semilla de longitud seq_len

    # Generar el vector de ruido para el generador
    noise = tf.random.normal([1, latent_dim])

    # Crear el escenario con el generador usando la semilla inicial
    generated_scenario = generator(noise, training=False).numpy().flatten()

    # Agregar el escenario generado a la lista de escenarios
    scenarios.append(generated_scenario)

# Convertir escenarios en un arreglo de NumPy para facilitar el backtesting
scenarios_array = np.array(scenarios)  # Shape: (100, 252)
