import yfinance as yf
import numpy as np
import pandas as pd


# Descargar los datos de precios históricos
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]
    return data

# Calcular rendimientos logarítmicos
def calculate_returns(data):
    returns = np.log(data / data.shift(1)).dropna()
    return returns

# Normalizar los rendimientos --- (datos -media)/ desviacion estandar
def normalize_returns(returns):
    mean = returns.mean()
    std_dev = returns.std()
    returns_norm = (returns - mean) / std_dev
    return returns_norm

# Función para obtener x_train_norm
def get_x_train_norm(ticker='AAPL', start_date='2014-11-01', end_date='2024-11-01'):
    data = download_data(ticker, start_date, end_date)
    returns = calculate_returns(data)
    returns_norm = normalize_returns(returns)
    x_train_norm = returns_norm.values  # Convertimos a valores numpy para el GAN
    return x_train_norm





# Desnormalización de los rendimientos generados
def denormalize_returns(normalized_returns, returns):
    mean = returns.mean()
    std_dev = returns.std()
    return (normalized_returns * std_dev) + mean







def plot_candlestick_with_signals(data, signals):
    # Asegúrate de tener 'Open', 'High', 'Low' y 'Close' en `data`
    ohlc_data = data[['Open', 'High', 'Low', 'Close']]

    # Crear listas de índices para las señales de compra/venta
    buy_signals = signals[signals['positions'] == 1].index
    sell_signals = signals[signals['positions'] == -1].index

    # Crear listas de precios para las señales
    buy_prices = data.loc[buy_signals, 'Close']
    sell_prices = data.loc[sell_signals, 'Close']

    # Añadir las señales de compra (verde) y venta (rojo)
    add_signals = [
        mpf.make_addplot(buy_prices, type='scatter', marker='^', color='green', markersize=100),
        mpf.make_addplot(sell_prices, type='scatter', marker='v', color='red', markersize=100)
    ]

    # Graficar con las señales de trading
    mpf.plot(ohlc_data, type='candle', style='charles', addplot=add_signals,
             title='Evolución del precio con señales de trading', ylabel='Precio')

    #
