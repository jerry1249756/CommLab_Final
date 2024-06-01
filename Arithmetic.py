from decimal import *
import sounddevice as sd
import soundfile as sf
import time
import numpy as np
import sys

class ArithmeticEncoding:
    def __init__(self, _sequence, _nlevels, _windows):
        self.data = _sequence
        self.n_levels = _nlevels
        self.windows = _windows
        self.quantize()
        self.frequency_map = self.build_freq_map()
        self.probability_table = self.get_probability_table()
        
    def quantize(self):
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        self.quantized_data = np.round((self.data - self.data_min) / step_size ).astype(int)

    def build_freq_map(self):
        freq_map = {}
        for symbol in self.quantized_data:
            freq_map[symbol] = freq_map.get(symbol, 0) + 1
        return freq_map

    def get_probability_table(self):
        total_frequency = sum(list(self.frequency_map.values()))
        probability_table = {}
        for key, value in self.frequency_map.items():
            probability_table[key] = value/total_frequency
        return probability_table

    def get_encoded_value(self, last_stage_probs):
        last_stage_probs = list(last_stage_probs.values())
        last_stage_values = []
        for sublist in last_stage_probs:
            for element in sublist:
                last_stage_values.append(element)

        last_stage_min = min(last_stage_values)
        last_stage_max = max(last_stage_values)
        encoded_value = (last_stage_min + last_stage_max)/2

        return last_stage_min, last_stage_max, encoded_value

    def process_stage(self, stage_min, stage_max):
        stage_probs = {}
        stage_domain = stage_max - stage_min
        for term_idx in range(len(self.probability_table.items())):
            term = list(self.probability_table.keys())[term_idx]
            term_prob = Decimal(self.probability_table[term])
            cum_prob = term_prob * stage_domain + stage_min
            stage_probs[term] = [stage_min, cum_prob]
            stage_min = cum_prob
        return stage_probs
    
    def process_stage_binary(self, float_interval_min, float_interval_max, stage_min_bin, stage_max_bin):
        stage_mid_bin = stage_min_bin + "1"
        stage_min_bin = stage_min_bin + "0"

        stage_probs = {}
        stage_probs[0] = [stage_min_bin, stage_mid_bin]
        stage_probs[1] = [stage_mid_bin, stage_max_bin]

        return stage_probs

    def encode_binary(self, float_interval_min, float_interval_max):
        binary_code = None

        stage_min_bin = "0.0"
        stage_max_bin = "1.0"

        stage_probs = {}
        stage_probs[0] = [stage_min_bin, "0.1"]
        stage_probs[1] = ["0.1", stage_max_bin]
        
        while True:
            if float_interval_max < bin2float(stage_probs[0][1]):
                stage_min_bin = stage_probs[0][0]
                stage_max_bin = stage_probs[0][1]
            else:
                stage_min_bin = stage_probs[1][0]
                stage_max_bin = stage_probs[1][1]

            stage_probs = self.process_stage_binary(float_interval_min, float_interval_max, stage_min_bin, stage_max_bin)
            if (bin2float(stage_probs[0][0]) >= float_interval_min) and (bin2float(stage_probs[0][1]) < float_interval_max):
                binary_code = stage_probs[0][0]
                break
            elif (bin2float(stage_probs[1][0]) >= float_interval_min) and (bin2float(stage_probs[1][1]) < float_interval_max):
                binary_code = stage_probs[1][0]
                break
        return binary_code
    
    def _encode(self, data):
        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)
        msg = data

        for msg_term_idx in range(len(msg)):
            stage_probs = self.process_stage(stage_min, stage_max)

            msg_term = msg[msg_term_idx]
            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

        last_stage_probs = self.process_stage(stage_min, stage_max)
        interval_min_value, interval_max_value, encoded_msg = self.get_encoded_value(last_stage_probs)

        return interval_min_value, interval_max_value
    
    def encode(self, windows):
        encoded = []
        i = 0
        while i < (len(self.quantized_data) // windows):
            interval_min_value, interval_max_value = self._encode(self.quantized_data[i * windows : (i + 1) * windows])
            binary_code = self.encode_binary(interval_min_value, interval_max_value)
            encoded.append(binary_code)
            i += 1
        interval_min_value, interval_max_value = self._encode(self.quantized_data[i * windows:])
        binary_code = self.encode_binary(interval_min_value, interval_max_value)
        encoded.append(binary_code)
        return encoded


    def _decode(self, enc_msg, msg_length):
        encoded_msg = bin2float(enc_msg)
        decoded_msg = []

        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)

        for idx in range(msg_length):
            stage_probs = self.process_stage(stage_min, stage_max)

            for msg_term, value in stage_probs.items():
                if encoded_msg >= value[0] and encoded_msg <= value[1]:
                    break

            decoded_msg.append(msg_term)

            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

        return decoded_msg
    
    def decode(self, encoded_msg, windows):
        decoded = []
        for code in encoded_msg[0 : len(encoded_msg) - 1]:
            decoded += self._decode(code, windows)
        decoded += self._decode(encoded_msg[-1], len(self.quantized_data) % windows)
        return decoded
    
    def test(self, fs = None):
        enc_start = time.time()
        encoded = self.encode(self.windows)
        enc_end = time.time()
        dec_start = time.time()
        decoded = self.decode(encoded, self.windows)
        dec_end = time.time()
        enc_len = 0
        for string in encoded:
            enc_len = enc_len + (len(string) - 2)
        self.prob_dist = np.bincount(self.quantized_data, minlength=self.n_levels)
        temp = self.prob_dist[self.prob_dist > 0] / len(self.data)
        print("============= Summary =============")
        print(f"Number of Input Symbols: {len(self.data)}")
        print(f"Number of Input Bits: {len(self.data) * np.log2(self.n_levels)}")
        print(f"Number of Encoded Bits: {enc_len}")
        print(f"Entropy: {-np.sum(temp * np.log2(temp))}")
        print(f"Encoding Time: {enc_end - enc_start}")
        print(f"Decoding Time: {dec_end - dec_start}")
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        decoded_data = np.array(decoded)*step_size + self.data_min
        sd.play(decoded_data, fs)
        sd.wait()

def float2bin(float_num, num_bits=None):
    float_num = str(float_num)
    if float_num.find(".") == -1:
        integers = float_num
        decimals = ""
    else:
        integers, decimals = float_num.split(".")
    decimals = "0." + decimals
    decimals = Decimal(decimals)
    integers = int(integers)

    result = ""
    num_used_bits = 0
    while True:
        mul = decimals * 2
        int_part = int(mul)
        result = result + str(int_part)
        num_used_bits = num_used_bits + 1

        decimals = mul - int(mul)
        if type(num_bits) is type(None):
            if decimals == 0:
                break
        elif num_used_bits >= num_bits:
            break
    if type(num_bits) is type(None):
        pass
    elif len(result) < num_bits:
        num_remaining_bits = num_bits - len(result)
        result = result + "0"*num_remaining_bits

    integers_bin = bin(integers)[2:]
    result = str(integers_bin) + "." + str(result)
    return result

def bin2float(bin_num):
    if bin_num.find(".") == -1:
        integers = bin_num
        decimals = ""
    else:
        integers, decimals = bin_num.split(".")
    result = Decimal(0.0)

    for idx, bit in enumerate(integers):
        if bit == "0":
            continue
        mul = 2**idx
        result = result + Decimal(mul)

    for idx, bit in enumerate(decimals):
        if bit == "0":
            continue
        mul = Decimal(1.0)/Decimal((2**(idx+1)))
        result = result + mul
    return result

if __name__ == '__main__':
    data, fs = sf.read('handel.wav')
    AC = ArithmeticEncoding(data, 64, 3)
    AC.test(fs)