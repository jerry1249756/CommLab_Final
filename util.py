import numpy as np

def epsilon_sequence(epsilon, length, n_symbols):
    symbols = list(range(n_symbols))
    high_prob_symbol = np.random.choice(symbols)
    prob = [(1 - epsilon) / (n_symbols - 1)] * n_symbols
    prob[high_prob_symbol] = epsilon
    
    sequence = np.random.choice(symbols, size=length, p=prob)
    
    return sequence