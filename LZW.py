import numpy as np
import time
import soundfile as sf
import sounddevice as sd
class LZW:
    def __init__(self, _sequence, _nlevels):
        self.data = _sequence
        self.n_levels = _nlevels
        self.quantize()
        self.initialize_dictionary()

    def quantize(self):
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        # quantize
        self.quantized_data = np.round((self.data - self.data_min) / step_size ).astype(int)

    def initialize_dictionary(self):
        self.dictionary = {}
        cnt_dict = 0
        for symbol in self.quantized_data:
            if tuple([symbol]) not in self.dictionary:
                self.dictionary[tuple([symbol])] = cnt_dict
                cnt_dict += 1

    def encode(self):
        cnt_dict = len(self.dictionary)
        S = [self.quantized_data[0]]
        encoded_sequence = []
        for cnt_seq in range(1, len(self.quantized_data)):
            C = [self.quantized_data[cnt_seq]]
            W = S + C
            if tuple(W) in self.dictionary:
                S = W + []
            else:
                encoded_sequence += [self.dictionary[tuple(S)]]
                self.dictionary[tuple(W)] = cnt_dict
                cnt_dict += 1
                S = C + []
        W = S + C
        encoded_sequence += [self.dictionary[tuple(S)]]
        self.complete_dict_len = len(self.dictionary)
        return encoded_sequence
    
    def decode(self, encoded_sequence):
        cnt_dict = len(self.dictionary)
        temp_dict = {y: x for x, y in self.dictionary.items()}
        OLD = encoded_sequence[0]
        NEW = -1
        C = []
        decoded_sequence = list(temp_dict[OLD])
        for symbol in encoded_sequence[1:]:
            NEW = symbol
            #print(temp_dict)
            if temp_dict.get(NEW, 0) == 0:
                S = list(temp_dict[OLD])
                S = S + C
            else:
                S = list(temp_dict[NEW])
            decoded_sequence += S
            C = [S[0]]
            temp_dict[cnt_dict] = tuple(list(temp_dict[OLD]) + C)
            cnt_dict += 1
            OLD = NEW
        return decoded_sequence
    
    def test(self, fs = None):
        self.initialize_dictionary()
        enc_start = time.time()
        encoded = self.encode()
        enc_end = time.time()
        self.initialize_dictionary()
        dec_start = time.time()
        decoded = self.decode(encoded)
        dec_end = time.time()
        self.prob_dist = np.bincount(self.quantized_data, minlength=self.n_levels)
        temp = self.prob_dist[self.prob_dist > 0] / len(self.data)
        print("============= Summary =============")
        print(f"Number of Input Symbols: {len(self.data)}")
        print(f"Number of Input Bits: {len(self.data) * np.log2(self.n_levels)}")
        print(f"Number of Encoded Bits: {len(encoded) * np.ceil(np.log2(self.complete_dict_len))}")
        print(f"Entropy: {-np.sum(temp * np.log2(temp))}")
        print(f"Encoding Time: {enc_end - enc_start}")
        print(f"Decoding Time: {dec_end - dec_start}")
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        decoded_data = np.array(decoded)*step_size + self.data_min
        sd.play(decoded_data, fs)
        sd.wait()

def epsilon_sequence(epsilon, length, n_symbols):
    symbols = list(range(n_symbols))
    high_prob_symbol = np.random.choice(symbols)
    prob = [(1 - epsilon) / (n_symbols - 1)] * n_symbols
    prob[high_prob_symbol] = epsilon
    sequence = np.random.choice(symbols, size=length, p=prob)
    return sequence

if __name__ == '__main__':
    data = epsilon_sequence(0.99, 1000000, 20)
    # data, fs = sf.read('handel.wav')
    lzw = LZW(data, 20)
    lzw.test()