import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
X, y = make_classification(n_samples=300,n_features=2,n_informative=2,n_redundant=0,n_classes=3,n_clusters_per_class=1,random_state=1)
model = MLPClassifier(hidden_layer_sizes=(2, 2),activation='identity',        solver='sgd',learning_rate_init=0.01,max_iter=1000,random_state=1)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = model.score(X, y)
print("Learning Rate:", 0.01)
print("Activation:", "Linear")
print("Hidden Layers:", 2)
print("Hidden Neurons:", 2)
print("Number of Classes:", 3)
print("Accuracy:", accuracy)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Neural Network Analysis of Multi-Class Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
