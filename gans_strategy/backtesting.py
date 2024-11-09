import pandas as pd
import ta  # Asegúrate de tener instalada esta librería

def passive_strategy(prices, initial_balance=10000):
    positions = initial_balance / prices['Close'][0]  # Comprar al inicio
    final_balance = positions * prices['Close'].iloc[-1]  # Valor al final
    return final_balance

# Función para crear señales de compra/venta basadas en el RSI
def create_signals(data: pd.DataFrame, column_name: str, **kwargs):
    data = data.copy()
    # Calcular el RSI con la ventana proporcionada
    rsi = ta.momentum.RSIIndicator(close=data[column_name], window=kwargs["rsi_window"]).rsi()
    data["RSI"] = rsi

    # Ajustar los umbrales de compra y venta para generar señales
    data["BUY_SIGNAL"] = data["RSI"] < kwargs["rsi_lower_threshold"]
    data["SELL_SIGNAL"] = data["RSI"] > kwargs["rsi_upper_threshold"]

    # Imprimir el número de señales de compra y venta generadas
    print(f"{column_name} BUY_SIGNALS: {data['BUY_SIGNAL'].sum()}")
    print(f"{column_name} SELL_SIGNALS: {data['SELL_SIGNAL'].sum()}")

    # Incluir la columna de precios original junto con las señales
    return data[[column_name, "RSI", "BUY_SIGNAL", "SELL_SIGNAL"]]

# Max Drawdown
def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown

# Win-Loss Ratio
def calculate_win_loss_ratio(trades):
    wins = sum(1 for trade in trades if trade['profit'] > 0)
    losses = sum(1 for trade in trades if trade['profit'] <= 0)
    return wins / losses if losses > 0 else float('inf')

# Sharpe Ratio
def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.00):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - (risk_free_rate / 252)
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

# Función de backtesting
def run_backtest(data, best_params, column_name):
    capital = INITIAL_CAPITAL
    n_shares = best_params["n_shares"]
    stop_loss = best_params["stop_loss"]
    take_profit = best_params["take_profit"]
    trades = []
    portfolio_value = [capital]
    active_positions = []

    # Generar señales para la columna específica
    technical_data = create_signals(data, column_name=column_name,
                                    rsi_window=best_params["rsi_window"],
                                    rsi_lower_threshold=best_params["rsi_lower_threshold"],
                                    rsi_upper_threshold=best_params["rsi_upper_threshold"])

    for i, row in technical_data.iterrows():
        # Cerrar posiciones basadas en TP o SL
        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            if pos["type"] == "LONG":
                if row[column_name] < pos["stop_loss"]:
                    capital += row[column_name] * pos["n_shares"] * (1 - COMMISSION)
                    trades.append({"type": "LONG", "profit": (row[column_name] - pos["bought_at"]) * pos["n_shares"]})
                    active_positions.remove(pos)
                elif row[column_name] > pos["take_profit"]:
                    capital += row[column_name] * pos["n_shares"] * (1 - COMMISSION)
                    trades.append({"type": "LONG", "profit": (row[column_name] - pos["bought_at"]) * pos["n_shares"]})
                    active_positions.remove(pos)

        # Abrir nuevas posiciones basadas en señales
        if row["BUY_SIGNAL"] and capital >= row[column_name] * n_shares * (1 + COMMISSION):
            capital -= row[column_name] * n_shares * (1 + COMMISSION)
            active_positions.append({
                "type": "LONG",
                "bought_at": row[column_name],
                "n_shares": n_shares,
                "stop_loss": row[column_name] * (1 - stop_loss),
                "take_profit": row[column_name] * (1 + take_profit)
            })

        # Actualizar valor del portafolio
        positions_value = sum(pos["n_shares"] * row[column_name] for pos in active_positions)
        portfolio_value.append(capital + positions_value)

    # Cerrar posiciones restantes
    for pos in active_positions.copy():
        capital += pos["n_shares"] * technical_data.iloc[-1][column_name] * (1 - COMMISSION)
        active_positions.remove(pos)

    portfolio_value.append(capital)
    max_drawdown = calculate_max_drawdown(portfolio_value)
    win_loss_ratio = calculate_win_loss_ratio(trades)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_value)

    return capital, max_drawdown, win_loss_ratio, sharpe_ratio, portfolio_value
