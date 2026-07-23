from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
X,y = make_regression(
n_samples=500,
n_features=1,
noise=10)
model = SGDRegressor(max_iter=1000)
model.fit(X,y)
print(model.score(X,y))
