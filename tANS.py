import numpy as np
import math
import copy

class tANS:
    def __init__(self, symbol_frequencies):
        self.symbol_frequencies = symbol_frequencies
        self.state_num = 256
        self.table_size = len(symbol_frequencies)
        self.build_tables()

    def build_tables(self):
      # Preparation
        temp1 = np.insert(self.symbol_frequencies, 0, 0)
        temp2 = np.cumsum(temp1)
        self.start = np.array(temp2[:-1]) - np.array(self.symbol_frequencies)
        # print(self.start)

        sum_freq = np.sum(self.symbol_frequencies)
        self.k = np.ceil(np.log2(sum_freq / self.symbol_frequencies)).astype(int)
        # print(self.k)

        power_of_2 = np.power(2, self.k)
        self.bound = np.array(self.symbol_frequencies) * power_of_2
        # print(self.bound)
        next = self.symbol_frequencies

      # StateTable
        self.state_table = [0 for _ in range(self.state_num)]
        step = ((self.state_num >> 1) + (self.state_num >> 3) + 3)
        pos = 0
        for s in range(self.table_size):
            for _ in range(self.symbol_frequencies[s]):
                self.state_table[pos] = s
                pos = (pos + step) % self.state_num
        # print(self.state_table)

      # EncodingTable
        self.encoding_table = [0 for _ in range(self.state_num)]
        next = copy.deepcopy(self.symbol_frequencies)
        for i in range(self.state_num):
          s = self.state_table[i]
          self.encoding_table[self.start[s] + next[s]] = i + self.state_num
          next[s] = next[s] + 1
          # print(self.symbol_frequencies)

        # print(self.encoding_table)
      # DecodingTable
        self.decoding_table = [[0, 0, 0] for _ in range(self.state_num)]
        next = copy.deepcopy(self.symbol_frequencies)
        for X in range(self.state_num):
          s = self.state_table[X]
          x = next[s]
          next[s] = next[s] + 1
          nbBits = np.log2(self.state_num).astype(int) - np.floor(np.log2(x)).astype(int)
          # print(x, np.floor(np.log2(x)).astype(int))
          newX = (x * pow(2, nbBits)) - self.state_num
          self.decoding_table[X] = [s, nbBits, newX]
        # print(self.decoding_table)

    def encode(self, input_symbols):
        x = self.state_num
        bit_string = ''
        self.symbol_length = len(input_symbols)
        for s in (input_symbols):
          # print(s)
          nbBits = self.k[s] - (x < self.bound[s])
          x_bin = bin(x)[2:]
          # print(x_bin[-nbBits:])
          bit_string += x_bin[-nbBits:]
          x = self.encoding_table[self.start[s] + (x >> nbBits)]
          # print(bit_string)

        self.final_state = x


        return bit_string, self.final_state

    def decode(self, encoded_bits, final_state):
        
        x = final_state
        decoded_symbols = []
        for _ in range(self.symbol_length):
          # print(x)
          t = self.decoding_table[x - self.state_num]
          # print(t)
          decoded_symbols.insert(0, t[0])
          
          # print(encoded_bits)
          if(t[1] == 0):
            getBit = '0'
          else:
            getBit = encoded_bits[-t[1]:]
            encoded_bits = encoded_bits[:-t[1]]

          x = t[2] + int(getBit, 2)

        return decoded_symbols
        
        
        
# Example usage
symbol_frequencies = [1, 1, 2, 3, 7, 17, 34, 63, 63, 34, 17, 7, 3, 2, 1, 1]
tans = tANS(symbol_frequencies)

input_symbols = [15, 14, 9, 8, 9, 8, 8, 9]
encoded_bit, final_state = tans.encode(input_symbols)
print(encoded_bit, final_state)

decoded_symbols = tans.decode(encoded_bit, final_state)
print(decoded_symbols)