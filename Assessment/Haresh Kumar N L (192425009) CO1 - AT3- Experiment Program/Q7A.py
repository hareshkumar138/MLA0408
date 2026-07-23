import numpy as np
x = np.array([1,2])
W = np.array([[0.5,0.2],
              [0.3,0.8]])
b = np.array([0.1,0.2])
z = np.dot(x,W)+b
output = 1/(1+np.exp(-z))
print(output)
