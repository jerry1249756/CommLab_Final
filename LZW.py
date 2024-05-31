class LZW:
    def __init__(self, sequence):
        self.sequence = sequence
        self.dictionary = {}

    def initialize_dictionary(self):
        self.dictionary = {}
        cnt_dict = 0
        for symbol in self.sequence:
            if tuple([symbol]) not in self.dictionary:
                self.dictionary[tuple([symbol])] = cnt_dict
                cnt_dict += 1

    def encode(self, original_sequence):
        self.initialize_dictionary()
        cnt_dict = len(self.dictionary)
        S = [original_sequence[0]]
        C = [original_sequence[1]]
        encoded_sequence = []
        for cnt_seq in range(2, len(original_sequence)):
            W = S + C
            if tuple(W) in self.dictionary:
                S = W + []
            else:
                encoded_sequence += [self.dictionary[tuple(S)]]
                self.dictionary[tuple(W)] = cnt_dict
                cnt_dict += 1
                S = C + []
            C = [original_sequence[cnt_seq]]
        W = S + C
        if tuple(W) in self.dictionary:
            encoded_sequence += [self.dictionary[tuple(W)]]
        else:
            encoded_sequence += [self.dictionary[tuple(S)]]
            encoded_sequence += [self.dictionary[tuple(C)]]
        
        return encoded_sequence
    
    def decode(self, encoded_sequence):
        self.initialize_dictionary()
        cnt_dict = len(self.dictionary)
        temp_dict = {y: x for x, y in self.dictionary.items()}
        S = []
        W = []
        decoded_sequence = []
        for symbol in encoded_sequence:
            W = list(temp_dict[symbol])
            decoded_sequence += W
            if len(S) > 0:
                temp_dict[cnt_dict] = S + [W[0]]
                cnt_dict += 1
            S = W + []

        return decoded_sequence
            