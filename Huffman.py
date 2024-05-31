class TreeNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

class HuffmanTree:
    def __init__(self, sequence):
        self.sequence = sequence
        self.freq_map = self.build_freq_map()
        self.root = self.build_huffman_tree()
        self.codes = self.generate_huffman_codes()

    def build_freq_map(self):
        freq_map = {}
        for symbol in self.sequence:
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

    def encode(self, original_sequence):
        encoded_sequence = []
        for symbol in original_sequence:
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
    
if __name__ == '__main__':
    sequence = [0, 1, 5, 9, 3, 5, 8]
    huffman_tree = HuffmanTree(sequence)
    encoded_sequence = huffman_tree.encode(sequence)
    decoded_sequence = huffman_tree.decode(encoded_sequence)
    print(encoded_sequence)
    print(decoded_sequence)
    