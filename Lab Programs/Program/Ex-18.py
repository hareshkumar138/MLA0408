import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
X, y = make_circles(n_samples=300, noise=0.1, factor=0.5)
model = MLPClassifier(hidden_layer_sizes=(2, 2),activation='tanh',solver='sgd',learning_rate_init=0.1,max_iter=1000,random_state=1)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = model.score(X, y)
print("Learning Rate:", 0.1)
print("Activation:", "Tanh")
print("Hidden Layers:", 2)
print("Hidden Neurons:", 2)
print("Accuracy:", accuracy)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Neural Network Analysis of Circular Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
