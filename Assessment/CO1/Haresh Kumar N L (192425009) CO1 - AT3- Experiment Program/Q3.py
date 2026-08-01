from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
iris = load_iris()
X_train,X_test,Y_train,Y_test = train_test_split(
    iris.data,iris.target,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train,Y_train)
pred = model.predict(X_test)
print("Accuracy =",accuracy_score(Y_test,pred))
print(confusion_matrix(Y_test,pred))
