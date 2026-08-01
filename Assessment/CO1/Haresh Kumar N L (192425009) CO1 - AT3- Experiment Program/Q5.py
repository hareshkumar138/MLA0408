import numpy as np
x = np.array([-2,-1,0,1,2])
sigmoid = 1/(1+np.exp(-x))
relu = np.maximum(0,x)
print("Input :",x)
print("Sigmoid :",sigmoid)
print("ReLU :",relu)
