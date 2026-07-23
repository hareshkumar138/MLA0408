from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
X,y = make_classification(
n_samples=200,
n_features=2,
n_redundant=0,
n_clusters_per_class=1,
random_state=1)
X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2)
model = Perceptron()
model.fit(X_train,y_train)
print("Accuracy =",model.score(X_test,y_test))
