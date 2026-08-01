import numpy as np
import matplotlib.pyplot as plt
X = np.array([1,2,3,4,5])
Y = np.array([2,4,6,8,10])
m = 0
c = 0
lr = 0.01
epochs = 1000
n = len(X)
loss = []
for i in range(epochs):
    y_pred = m*X + c
    cost = np.mean((Y-y_pred)**2)
    loss.append(cost)
    dm = (-2/n)*np.sum(X*(Y-y_pred))
    dc = (-2/n)*np.sum(Y-y_pred)
    m -= lr*dm
    c -= lr*dc
print("Slope =",m)
print("Intercept =",c)
plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.show()
