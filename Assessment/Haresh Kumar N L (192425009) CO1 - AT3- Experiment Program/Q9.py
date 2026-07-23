from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
X,y = make_regression(
n_samples=500,
n_features=10)
model = MLPRegressor(
batch_size=32,
max_iter=300)
model.fit(X,y)
print("Training Completed")
