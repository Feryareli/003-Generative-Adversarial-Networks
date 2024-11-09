import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm
from utils.utils import download_data, calculate_returns, normalize_returns, denormalize_returns
from gans_strategy.train_gan import build_generator, build_discriminator, train_step
from gans_strategy.backtesting import passive_strategy, create_signals, calculate_max_drawdown, calculate_win_loss_ratio, calculate_sharpe_ratio, run_backtest
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

# DATA FRAME DE LAS SIMULACIONES
# Supongamos que tienes la serie original y el precio inicial
original_data = data.Close.AAPL.values[:252]  # Serie original de precios
initial_price = original_data[0]  # Precio inicial de 2024

# Crear una lista para almacenar cada serie simulada
num_scenarios = 100  # Número de escenarios que quieres simular
simulated_paths = []

for i in range(num_scenarios):
    # Desnormalizar los retornos generados para cada escenario
    denormalized_returns = denormalize_returns(scenarios_array[i], mean_return, std_dev_return)
    # Acumular rendimientos para simular precios y partir desde el precio inicial
    simulated_path = initial_price * np.exp(np.cumsum(denormalized_returns))
    simulated_paths.append(simulated_path)

# Crear un diccionario para las columnas del DataFrame
data_dict = {
    'Fecha': pd.date_range(start='2024-01-01', periods=len(original_data), freq='B'),  # Fechas de días hábiles
    'Serie Original': original_data
}

# Añadir cada serie simulada al diccionario
for i in range(num_scenarios):
    data_dict[f'Serie Simulada {i+1}'] = simulated_paths[i]

# Crear el DataFrame con todas las series
df = pd.DataFrame(data_dict)

# Establecer la columna de fechas como índice (opcional)
df.set_index('Fecha', inplace=True)

# Cargar el dataset
df = df

#Parámetros iniciales
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.00125



# Parámetros óptimos de ejemplo
best_params =[ {
    "n_shares": 81,
    "stop_loss": 0.08,
    "take_profit": 0.20,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.05,
    "take_profit": 0.15,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.06,
    "take_profit": 0.18,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.10,
    "take_profit": 0.25,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.08,
    "take_profit": 0.28,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.15,
    "take_profit": 0.25,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.09,
    "take_profit": 0.27,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.5,
    "take_profit": 0.17,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.095,
    "take_profit": 0.2156,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
},
{
    "n_shares": 81,
    "stop_loss": 0.143,
    "take_profit": 0.2546,
    "rsi_window": 10,
    "rsi_lower_threshold": 90,
    "rsi_upper_threshold": 99
}

]

for i in best_params:
    # Ejecutar el backtest en "Serie Original" y todas las series simuladas
    for column_name in df.columns:
        if column_name == "Serie Original" or column_name.startswith("Serie Simulada"):
            print(f"Running backtest on {column_name}...")
            final_capital, max_drawdown, win_loss_ratio, sharpe_ratio, portfolio_values = run_backtest(df, i,
                                                                                                       column_name)

            initial_capital = INITIAL_CAPITAL
            pnl = final_capital - initial_capital
            annual_return = (final_capital / initial_capital) ** (252 / len(portfolio_values)) - 1
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else float('inf')

            print(f"Results for {column_name}:")
            print(f"Final Portfolio Value: ${final_capital:,.2f}")
            print(f"P&L: ${pnl:,.2f}")
            print(f"Max Drawdown (%): {max_drawdown * 100:.2f}")
            print(f"Win-Loss Ratio: {win_loss_ratio:.2f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Calmar Ratio: {calmar_ratio:.2f}\n")

