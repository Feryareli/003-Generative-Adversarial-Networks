import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm
from utils.utils import download_data, calculate_returns, normalize_returns
from gans_strategy.train_gan import build_generator, build_discriminator
from gans_strategy.backtest import passive_strategy
from gans_strategy.strategy import rsi_trading_strategy
from utils.utils import get_x_train_norm

# Configuración de los parámetros del proyecto
ticker = 'AAPL'
start_date = '2014-11-01'
end_date = '2024-11-01'

# Obtener x_train_norm
x_train_norm = get_x_train_norm(ticker='AAPL', start_date='2014-11-01', end_date='2024-11-01')

latent_dim = 300
num_scenarios = 100
seq_len = 252

# Crear (instanciar) el modelo generador
generator = build_generator(latent_dim=latent_dim, seq_len=seq_len)

# Crear (instanciar) el modelo discriminador
discriminator = build_discriminator(seq_len=seq_len)

# Ver el resumen del generador (opcional)
generator.summary()

# Entrenar el GAN
gen_loss_history, disc_loss_history = train_gan(generator, discriminator, x_train_norm)


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
scenarios_array




import matplotlib.pyplot as plt

# Generar datos simulados de scenarios_array para 100 escenarios, cada uno con 252 puntos
num_scenarios = 100
seq_len = 252

# Graficar algunos de los escenarios sinteticos
plt.figure(figsize=(12, 6))
for i in range(min(num_scenarios, 100)):  # Graficamos solo 100 escenarios para mayor claridad
    plt.plot(scenarios_array[i], alpha=0.7, label=f"Escenario {i+1}")

plt.title("Rendimientos Sintéticos Generados por el GAN")
plt.xlabel("Días")
plt.ylabel("Rendimiento")
plt.show()






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





stop_loss = 0.07
take_profit = 0.11

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})

sum(signals.positions == 1)


stop_loss = 0.01
take_profit = 0.07

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})





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



stop_loss = 0.01
take_profit = 0.07

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})


stop_loss = 0.02
take_profit = 0.01

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})

stop_loss = 0.02
take_profit = 0.02

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

stop_loss = 0.03
take_profit = 0.01

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})



stop_loss = 0.03
take_profit = 0.02

profit_scenarios = []
for scenario in scenarios_array:
    simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
    signals = simple_trading_strategy(simulated_prices)
    final_balance = backtest_strategy(simulated_prices, signals, stop_loss, take_profit)
    profit_scenarios.append(final_balance)

avg_profit = np.mean(profit_scenarios)
print({'stop_loss': stop_loss, 'take_profit': take_profit, 'avg_profit': avg_profit})






stop_loss = 0.03
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







# Lista de resultados
results = []

for sl in stop_loss_levels:
    for tp in take_profit_levels:
        profit_scenarios = []

        for scenario in scenarios_array:
            simulated_prices = pd.DataFrame(np.cumsum(scenario), columns=['Close']) + original_data[-1]
            signals = simple_trading_strategy(simulated_prices)
            final_balance = backtest_strategy(simulated_prices, signals, sl, tp)
            profit_scenarios.append(final_balance)

        # Calcular métricas
        avg_profit = np.mean(profit_scenarios)
        sharpe_ratio, calmar_ratio, max_drawdown = calculate_metrics(profit_scenarios)

        # Añadir resultados a la lista
        results.append({
            'Stop-Loss': sl,
            'Take-Profit': tp,
            'Average Profit': avg_profit,
            'Sharpe Ratio': sharpe_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown
        })

# Crear el DataFrame con todas las columnas
results_df = pd.DataFrame(results)

# Visualizar la tabla en el notebook
results_df








# Ordenar
optimal_strategy = results_df.sort_values(by='Calmar Ratio', ascending=False).iloc[0]
print("Mejores parámetros según el ratio de Calmar:", optimal_strategy)


# Calcular el valor final de la estrategia activa (óptima) usando el mejor Stop-Loss y Take-Profit
signals_optimal = simple_trading_strategy(data)  # Genera señales usando la estrategia activa
final_value_active = backtest_strategy(data, signals_optimal, optimal_strategy['Stop-Loss'], optimal_strategy['Take-Profit'])

# Calcular el valor final de la estrategia pasiva (compra y retención)
final_value_passive = passive_strategy(data)  # Valor final de la estrategia pasiva

# Crear el gráfico comparativo
plt.figure(figsize=(10, 6))
plt.bar(['Estrategia Activa', 'Estrategia Pasiva'], [final_value_active, final_value_passive], color=['blue', 'grey'])
plt.title("Comparación de Rendimiento: Estrategia Activa vs. Pasiva")
plt.ylabel("Valor Final del Portafolio")
plt.show()



optimal_metrics = pd.DataFrame({
    'Stop-Loss': [optimal_strategy['stop_loss']],
    'Take-Profit': [optimal_strategy['take_profit']],
    'Average Profit': [optimal_strategy['avg_profit']],
    'Sharpe Ratio': [optimal_strategy['sharpe_ratio']],
    'Calmar Ratio': [optimal_strategy['calmar_ratio']],
    'Max Drawdown': [optimal_strategy['max_drawdown']]
})

optimal_metrics