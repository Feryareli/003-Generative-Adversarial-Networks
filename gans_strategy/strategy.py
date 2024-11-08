import pandas as pd
import numpy as np
import ta


def rsi_trading_strategy(prices, rsi_window=14, overbought=70, oversold=30):
    # Cálculo del RSI
    rsi = ta.momentum.rsi(close=prices['Close'], window=rsi_window, fillna=False)

    # Crear el DataFrame de señales y asignar utilizando `loc`
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0

    # Generar señales de compra y venta basadas en los niveles de sobrecompra y sobreventa del RSI
    signals.loc[prices.index[rsi_window:], 'signal'] = np.where(
        (rsi[rsi_window:] < oversold),  # Señal de compra cuando el RSI está por debajo del umbral de sobreventa
        1,
        np.where(
            (rsi[rsi_window:] > overbought),  # Señal de venta cuando el RSI está por encima del umbral de sobrecompra
            -1,
            0
        )
    )

    # Generar posiciones basadas en las señales
    signals['positions'] = signals['signal'].diff()
    return signals