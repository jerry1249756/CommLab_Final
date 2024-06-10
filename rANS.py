import sounddevice as sd
import soundfile as sf
import numpy as np
import sys
import time
from scipy.stats import entropy
from util import epsilon_sequence


class rANSEncoder:
    def __init__(self, _input, _nlevels) -> None:
        self.data = _input
        self.n_levels = _nlevels # quantize levels
        self.quantized_data = None
        self.encoded_state = None
        self.encoded_bitstream = []
        self.data_max = 0
        self.data_min = 0
        self.prob_dist = None
        self.cum_dist = None
        print("Using rANS Encoding...")
    
    def compute_statistics(self):
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        # quantize
        self.quantized_data = np.round((self.data - self.data_min) / step_size ).astype(int)

        # calculate symbol apperance count / cumulative count
        self.prob_dist = np.bincount(self.quantized_data, minlength=self.n_levels)
        cum_dist = np.cumsum(self.prob_dist)

        # Adding 0 at the beginning
        self.cum_dist = np.insert(cum_dist, 0, 0) 
    
    def compute_statistics_proposal(self, n):
        # compute the statistics using the proposal distribution with the first 2**n data.
        assert 2**n < len(self.data)

        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        # quantize
        self.quantized_data = np.round((self.data - self.data_min) / step_size ).astype(int)

        # calculate symbol apperance count / cumulative count BASE ON 2**N SYMBOLS!!!
        self.prob_dist = np.bincount(self.quantized_data[0:2**n-1], minlength=self.n_levels)
        cum_dist = np.cumsum(self.prob_dist)

        # Adding 0 at the beginning
        self.cum_dist = np.insert(cum_dist, 0, 0) 

        pass

    def calc_KL_divergence(self):
        est_prob_dist = self.prob_dist / sum(self.prob_dist)
        real_prob_dist = np.bincount(self.quantized_data, minlength=self.n_levels)
        real_prob_dist = real_prob_dist / sum(real_prob_dist)

        # D(Q||P), Q: target, P: proposal
        print(f"KL divergence: {entropy(real_prob_dist, est_prob_dist)}")
        return 
    



    def _encode_helper(self, state, symbol, tot_count):
        prob = self.prob_dist[symbol]
        cum_prob = self.cum_dist[symbol]
        next_state = int( (state//prob) * tot_count + cum_prob + state%prob)
        return next_state
    
    def encode(self):
        print("Encoding...")
        start = time.time()
        tot_counts = len(self.data)
        bit_stream = []
        state = tot_counts
        for symbol in self.quantized_data:
            while state >= 2*self.prob_dist[symbol]:
                bit_stream.append(state%2)
                state = state//2
            state = self._encode_helper(state, symbol, tot_counts)
            # print(f"state:{state}, symbol:{symbol}")

        self.encoded_state = state
        # self.encoded_bitstream = bit_stream
        self.encoded_bitstream = ''.join(map(str, bit_stream))
        end = time.time()

        temp = self.prob_dist[self.prob_dist>0] 
        temp = temp / sum(temp)
        print("============= Summary =============")
        print(f"Number of Input Symbols: {len(self.data)}")
        print(f"Final State: {self.encoded_state}")
        # print(f"Final Bitstream: {''.join(str(bit) for bit in self.encoded_bitstream)}")
        print(f"Average Codeword Length: { (len(self.encoded_bitstream) + len(bin(state)) - 2)/ len(self.data)}") # "0b...."
        print(f"Entropy: {-np.sum(temp * np.log2(temp))}")
        print(f"Processing time: {end-start} (s)")
        # print(f"Memory Usage: {self.getUsage() / 1024} (KB)")
    
    def getUsage(self):
        # get the usage of encoding in terms of bytes
        # usage = [self.encoded_state, self.encoded_bitstream, self.prob_dist]
        usage = [self.encoded_bitstream]
        mem_usage = 0
        for item in usage:
            mem_usage += sys.getsizeof(item)
        return mem_usage


    def _decode_helper(self, state, tot_count):
        slot = state % tot_count
        symbol = int(np.searchsorted(self.cum_dist, slot, side="right") -1)
        prob = self.prob_dist[symbol]
        cum_prob = self.cum_dist[symbol]
        prev_state = int(np.floor(state // tot_count)*prob + slot - cum_prob)
        # print(f"{state, symbol, prev_state}")
        return symbol, prev_state

    
    def decode(self):
        #note that we have reversed the input stream, so decoding can be in normal direction!
        print("Decoding...")
        print(f"Encoded state: {self.encoded_state}")
        start = time.time()
       
        self.encoded_bitstream = [int(bit) for bit in self.encoded_bitstream]
        temp = len(self.encoded_bitstream)
        tot_counts = len(self.data)
        decoded_symbols = []
        
        while len(decoded_symbols) < tot_counts:
            # if(len(self.encoded_bitstream)==0):
            #     print(state)
            symbol, state = self._decode_helper(self.encoded_state, tot_counts)
            # print(f"{(self.encoded_state, symbol)}")
           
            while state < tot_counts:
                bit = self.encoded_bitstream.pop()
                state = state*2 + bit
            decoded_symbols.append(symbol)
            
            self.encoded_state = state
        

        
        # symbol, state = self._decode_helper(self.encoded_state, tot_counts)
        # print(f"{symbol} {state}")
        # symbol, state = self._decode_helper(self.encoded_state, tot_counts)
        # print(f"{symbol} {state}")
        # symbol, state = self._decode_helper(self.encoded_state, tot_counts)
        # print(f"{symbol} {state}")
        decoded_symbols.reverse()

        step_size = (self.data_max - self.data_min) / (self.n_levels-1)
        decoded_data = np.array(decoded_symbols)*step_size + self.data_min
        end = time.time()
        print("============= Summary =============")
        print(f"Length of Decoding Sequence: {temp}")
        print(f"Processing time: {end-start} (s)")
        return decoded_data

        


# Example Usage

## Arbitrary list
# data = [0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,15,15,15,12,13,13,13,13,13,13,13,14,15,13]
## Audio file
data, fs = sf.read('handel.wav')
## A biased sequence where a specific symbol would occur with high probability
# data = epsilon_sequence(0.99,10000,20)

encoder = rANSEncoder(data, 32)
encoder.compute_statistics() # use all data to get accurate distribution
# encoder.compute_statistics_proposal(18) # use 2**n data to approximate distribution (maximum 18 for handel.wav)
temp = encoder.encode()
decoded_data = encoder.decode()
encoder.calc_KL_divergence()


sd.play(decoded_data, fs)
sd.wait() # wait until the play has finished