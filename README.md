# CommLab_Final

### Usage

```
data, fs = sf.read('handel.wav')
encoder = ANSEncoder(data)
encoded_data = encoder.encode()
mem_usage = encoder.getUsage()
run_time = encoder.getRuntime()

decoder = ANSDecoder(encoded_data) # depends on the encoded data format
decoded_data = decoder.decode()
mem_usage = decoder.getUsage()
run_time = decoder.getRuntime()
sd.play(decoded_data, fs)
# sd.wait() # wait until the play has finished
```
