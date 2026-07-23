import numpy as np
import matplotlib.pyplot as plt
x = 8
lr = 0.1
values=[]
for i in range(30):
    grad = 2*x
    x = x-lr*grad
    values.append(x)
plt.plot(values)
plt.show()
