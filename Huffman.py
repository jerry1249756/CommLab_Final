import numpy as np
import soundfile as sf
import sounddevice as sd
import time
class TreeNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

class HuffmanTree:
    def __init__(self, _sequence, _nlevels):
        self.data = _sequence
        self.n_levels = _nlevels
        self.quantize()
        self.freq_map = self.build_freq_map()
        self.root = self.build_huffman_tree()
        self.codes = self.generate_huffman_codes()

    def quantize(self):
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        # quantize
        self.quantized_data = np.round((self.data - self.data_min) / step_size ).astype(int)

    def build_freq_map(self):
        freq_map = {}
        for symbol in self.quantized_data:
            freq_map[symbol] = freq_map.get(symbol, 0) + 1
        return freq_map

    def build_huffman_tree(self):
        nodes = [TreeNode(symbol, freq) for symbol, freq in self.freq_map.items()]
        while len(nodes) > 1:
            nodes.sort(key=lambda x: x.freq)
            left = nodes.pop(0)
            right = nodes.pop(0)
            merged = TreeNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            nodes.append(merged)
        return nodes[0]

    def generate_huffman_codes(self):
        codes = {}
        def traverse(node, code=[]):
            if node:
                if node.symbol != None:
                    codes[node.symbol] = code
                traverse(node.left, code + [0])
                traverse(node.right, code + [1])
        traverse(self.root)
        return codes

    def encode(self):
        encoded_sequence = []
        for symbol in self.quantized_data:
            encoded_sequence += self.codes[symbol]
        return encoded_sequence

    def decode(self, encoded_sequence):
        decoded_sequence = []
        current_node = self.root
        for bit in encoded_sequence:
            if bit == 0:
                current_node = current_node.left
            else:
                current_node = current_node.right
            if not current_node.left and not current_node.right:
                decoded_sequence += [current_node.symbol]
                current_node = self.root
        return decoded_sequence
    
    def test(self, fs = None):
        enc_start = time.time()
        encoded = self.encode()
        enc_end = time.time()
        dec_start = time.time()
        decoded = self.decode(encoded)
        dec_end = time.time()
        self.prob_dist = np.bincount(self.quantized_data, minlength=self.n_levels)
        temp = self.prob_dist[self.prob_dist > 0] / len(self.data)
        print("============= Summary =============")
        print(f"Number of Input Symbols: {len(self.data)}")
        print(f"Number of Input Bits: {len(self.data) * np.log2(self.n_levels)}")
        print(f"Number of Encoded Bits: {len(encoded)}")
        print(f"Entropy: {-np.sum(temp * np.log2(temp))}")
        print(f"Encoding Time: {enc_end - enc_start}")
        print(f"Decoding Time: {dec_end - dec_start}")
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        decoded_data = np.array(decoded)*step_size + self.data_min
        sd.play(decoded_data, fs)
        sd.wait()


def epsilon_sequence(epsilon, length, n_symbols):
    prob = [epsilon] + [(1 - epsilon) / (n_symbols - 1)] * (n_symbols - 1)
    symbols = list(range(n_symbols))
    sequence = np.random.choice(symbols, size=length, p=prob)
    
    return sequence
if __name__ == '__main__':
    #data = epsilon_sequence(0.99, 100000, 20)
    data, fs = sf.read('handel.wav')
    huffman_tree = HuffmanTree(data, 64)
    huffman_tree.test(fs)
    