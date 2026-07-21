import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=100,n_features=2,n_informative=2, n_redundant=0,n_clusters_per_class=1, random_state=42)
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0],X[:, 1],c=y,cmap='viridis',edgecolors='black',s=70)
coef = model.coef_[0]
intercept = model.intercept_[0]
x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_vals = -(coef[0]*x_vals + intercept)/coef[1]
plt.plot(x_vals, y_vals, 'r--', linewidth=2, label='Decision Boundary')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
