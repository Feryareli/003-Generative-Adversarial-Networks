import mplfinance as mpf

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
