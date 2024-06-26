import sounddevice as sd
import soundfile as sf
import numpy as np
import copy
import time

from util import epsilon_sequence


class tANS:
    def __init__(self, _data, _level, _states):
        self.data = _data
        self.state_num = _states
        self.table_size = _level
        
        self.symbol_frequencies = self.compute_statistics()
        self.build_tables()
        
    def compute_statistics(self):
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        step_size = (self.data_max - self.data_min) / (self.table_size-1)
        # quantize
        self.quantized_data = np.round((self.data - self.data_min) / step_size ).astype(int)
        # print(self.quantized_data[:200])
        
        # calculate symbol apperance count / cumulative count
        self.prob_dist = np.bincount(self.quantized_data, minlength=self.table_size)
        # print(self.prob_dist)
        frequency = self.adjust_sum_to_power_of_2(self.prob_dist, self.state_num)
        # print(frequency)
        # print(f"Frequency Sum: {np.sum(frequency)}")
        # print(frequency)
        return frequency
        
    def adjust_sum_to_power_of_2(self, values, n):
        target_sum = n
        current_sum = np.sum(values)
        
        # Calculate the scaling factor
        scaling_factor = target_sum / current_sum
        
        # Scale the values and round to the nearest integer
        scaled_values_float = values * scaling_factor
        
        # Ensure all scaled values are at least 1 after rounding
        scaled_values = np.maximum(1, np.round(scaled_values_float)).astype(np.int64)
        
        # Calculate the difference
        difference = target_sum - np.sum(scaled_values)
        
        # Adjust the scaled values to sum to the target sum
        while difference != 0:
            if difference > 0:
                # Find index with maximum fractional part to increment
                fractional_parts = (scaled_values_float - scaled_values)
                idx = np.argmax(fractional_parts)
                scaled_values[idx] += 1
                difference -= 1
            else:
                # Find index with minimum fractional part to decrement, ensuring values remain at least 1
                fractional_parts = (scaled_values_float - scaled_values)
                idx = np.argmin(fractional_parts)
                if scaled_values[idx] > 1:
                    scaled_values[idx] -= 1
                    difference += 1
                else:
                    # Find next suitable index to decrement
                    for i in range(len(scaled_values)):
                        if scaled_values[i] > 1:
                            scaled_values[i] -= 1
                            difference += 1
                            break
        
        return scaled_values


    def build_tables(self):
      # Preparation
        temp1 = np.insert(self.symbol_frequencies, 0, 0)
        temp2 = np.cumsum(temp1)
        self.start = np.array(temp2[:-1]) - np.array(self.symbol_frequencies)
        # print(self.start)

        sum_freq = np.sum(self.symbol_frequencies)
        # print(sum_freq)
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

    def encode(self):
        print("Encoding...")
        start = time.time()
        x = self.state_num
        bit_string = ''
        self.symbol_length = len(self.quantized_data)
        # print(f"Total length is {self.symbol_length}")
        for s in (self.quantized_data):
          # print(s)
          nbBits = self.k[s] - (x < self.bound[s])
          x_bin = bin(x)[2:]
          # print(nbBits)
          # print(x_bin[-nbBits:])
          if(nbBits > 0):
            bit_string += x_bin[-nbBits:]
          x = self.encoding_table[self.start[s] + (x >> nbBits)]
          # print(bit_string)

        self.final_state = x
        # print(x)
        end = time.time()
        print("Finish Encoding.")
        
        temp = self.prob_dist[self.prob_dist>0] / sum(self.prob_dist)
        print("============= Summary =============")
        print(f"Number of Input Symbols: {self.symbol_length}")
        print(f"Total Codeword Length: { len(bit_string)}") # "0b...."
        print(f"Average Codeword Length: { len(bit_string)/ self.symbol_length}") # "0b...."
        print(f"Entropy: {-np.sum(temp * np.log2(temp))}")
        print(f"Processing time: {end-start} (s)")
        
        return bit_string

    def decode(self, encoded_bits):
        
        x = self.final_state
        decoded_symbols = []
        encoded_bits = [int(bit) for bit in encoded_bits]
        print("Decoding...")
        start = time.time()
        for _ in range(len(self.data)):
          # print(x)
          t = self.decoding_table[x - self.state_num]
          # print(t)
          decoded_symbols.append(t[0])
          
          # print(encoded_bits)
          if(t[1] == 0):
            getBit = [0]
          else:
            if(len(encoded_bits) == 0):
              getBit = [0]
            else: 
              getBit = encoded_bits[-t[1]:]
              for _ in range(t[1]):
                    encoded_bits.pop()
          
          getBit = ''.join([str(num) for num in getBit])
          x = t[2] + int(getBit, 2)

        decoded_symbols.reverse()
        step_size = (self.data_max - self.data_min) / (self.table_size-1)
        decoded_data = np.array(decoded_symbols)*step_size + self.data_min
        end = time.time()
        print("Finish Decoding.")
        print("============= Summary =============")
        print(f"Length of Decoding Sequence: {len(decoded_data)}")
        print(f"Processing time: {end-start} (s)")
        return decoded_data
      
    def test(self, fs = None):
        
        encoded_bits = self.encode()

        decoded_data = self.decode(encoded_bits)
        sd.play(decoded_data, fs)
        sd.wait()
        
# Example usage
# data, fs = sf.read('handel.wav')

# data = epsilon_sequence(0.95, 200000, 20)
# print(data[:200])
# tans = tANS(data, 32, 1024)
# tans.test()

# sd.play(decoded_data, fs)
# sd.wait()