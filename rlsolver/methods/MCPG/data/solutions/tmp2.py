
import numpy as np

s = np.load("SNR.npy")
b = np.isscalar(s)
s_shape_len = len(s.shape)
print(s_shape_len)
print(b)
print(s)

k = np.load("K.npy")
print("k", k)
k_shape_len = len(k.shape)
print(k_shape_len)
print()
