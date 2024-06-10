import numpy as np
import sounddevice as sd
import soundfile as sf

def epsilon_sequence(epsilon, length, n_symbols):
    symbols = list(range(n_symbols))
    high_prob_symbol = np.random.choice(symbols)
    prob = [(1 - epsilon) / (n_symbols - 1)] * n_symbols
    prob[high_prob_symbol] = epsilon
    
    sequence = np.random.choice(symbols, size=length, p=prob)
    
    return sequence

def calc_entropy_rate(sequence, n_symbols):
    # build the transition matrix
    transition_count = np.zeros((n_symbols, n_symbols))
    for i in range(len(sequence) - 1):
        current_symbol = sequence[i]
        next_symbol = sequence[i+1]
        transition_count[current_symbol, next_symbol] += 1
    
    transition_mat = transition_count / transition_count.sum(axis=1, keepdims=True)
    # replace the 0 prob rows into equal probability
    transition_mat[np.isnan(transition_mat)] = 1.0/n_symbols

    eigen_val, eigen_vec = np.linalg.eig(transition_mat.T)

    evec1 = eigen_vec[:,np.isclose(eigen_val, 1)]
    evec1 = evec1[:,0]

    stationary_dist = evec1 / evec1.sum()
    stationary_dist = stationary_dist.real
    # check stationary state
    # print(stationary_dist)
    # print(np.matmul(transition_mat.T, stationary_dist.T))

    entropy_rate = 0.0
    # H(X) = sum_{ij} u_i P_{ij} log P_{ij}, u: stationary dist, P: transition matrix
    for i in range(n_symbols):
        for j in range(n_symbols):
            if transition_mat[i, j] > 0:
                entropy_rate += stationary_dist[i] * transition_mat[i, j] * np.log2(transition_mat[i, j])
    
    entropy_rate = -entropy_rate
    return entropy_rate



# input_seq = [0,1,0,2,0,3,1,4,1,5,0,6,1,3,1,7,1,6,2,6,5,3,5,4,5,6,5,7]
# calc_entropy_rate(input_seq, 8)
data, fs = sf.read('handel.wav')

data_min = np.min(data)
data_max = np.max(data)
n_levels = 32
step_size = (data_max - data_min) / (n_levels-1)
# quantize
quantized_data = np.round((data - data_min) / step_size ).astype(int)
entropy_rate = calc_entropy_rate(quantized_data, n_levels)

print(entropy_rate)