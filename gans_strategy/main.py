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








# Generar fechas y datos ficticios para la serie original
dates = pd.date_range(start='2024-01-01', periods=seq_len, freq='B')  # Fechas de días hábiles para la simulación
original_data = data.Close.AAPL.values[:252]  # Datos de ejemplo acumulados como una serie de precios
initial_price = original_data[0]  # Precio inicial de 2024

# Graficar los escenarios simulados y el escenario original
plt.figure(figsize=(14, 7))

# Graficar 100 escenarios simulados de `scenarios_array`
for i in range(min(num_scenarios, 100)):  # Graficamos solo 100 escenarios para mayor claridad
    # Desnormalizar los rendimientos generados
    denormalized_returns = denormalize_returns(scenarios_array[i], mean_return, std_dev_return)

    # Acumular rendimientos para simular precios
    simulated_path = initial_price + np.exp(np.cumsum(denormalized_returns))  # Acumulamos rendimientos desnormalizados

    plt.plot(dates, simulated_path, alpha=0.5, linestyle="--", label=f"Escenario Simulado {i + 1}" if i < 1 else "")

# Graficar el escenario original
plt.plot(dates, original_data, color="black", linewidth=2, label="Serie Original")

# Configuraciones de la gráfica
plt.title("Comparación de la Serie de Precios Original vs Escenarios Simulados")
plt.xlabel("Fecha")
plt.ylabel("Precio Simulado")
plt.legend()
plt.show()







# Evaluación de la estrategia con niveles específicos de stop-loss y take-profit
stop_loss = 0.07
take_profit = 0.7

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})







# Evaluación de la estrategia con niveles específicos de stop-loss y take-profit
stop_loss = 0.02
take_profit = 0.03

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})






# Evaluación de la estrategia con niveles específicos de stop-loss y take-profit
stop_loss = 0.04
take_profit = 0.04

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})








# Ordenar
optimal_strategy = results_df.sort_values(by='Calmar Ratio', ascending=False).iloc[0]
print("Mejores parámetros según el ratio de Calmar:", optimal_strategy)
