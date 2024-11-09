import pandas as pd
def passive_strategy(prices, initial_balance=10000):
    positions = initial_balance / prices['Close'][0]  # Comprar al inicio
    final_balance = positions * prices['Close'].iloc[-1]  # Valor al final
    return final_balance
