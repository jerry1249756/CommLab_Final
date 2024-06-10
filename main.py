import sounddevice as sd
import soundfile as sf
import numpy as np
import sys
import time

from Huffman import HuffmanTree
from Arithmetic import ArithmeticEncoding
from rANS import rANSEncoder
from tANS import tANS
from LZW import LZW

## data read
data, fs = sf.read('handel.wav')

## Huffman Coding
print("Huffman Coding.")
print("Start.")
huffman_tree = HuffmanTree(data, 32)    # input: data, quantize level
huffman_tree.test(fs)

## Arithmetic Coding
print("Arithmetic Coding. (Consume Long Time, Can Reduce Quantize Level)")
print("Start.")
AC = ArithmeticEncoding(data, 32, 3)    # input: data, quantize level, windows
AC.test(fs)

## rANS
print("rANS.")
print("Start.")
rans = rANSEncoder(data, 32)            # input: data, quantize level
rans.test(fs)

## tANS
print("tANS.")
print("Start.")
tans = tANS(data, 32, 1024)             # input: data, quantize level, states
tans.test(fs)

## LZW
print("LZW.")
print("Start.")
lzw = LZW(data, 32)                     # input: data, quantize level
lzw.test()