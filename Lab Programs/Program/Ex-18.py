import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X, y = make_classification(n_samples=500,n_features=2,n_informative=2,n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential([Dense(3, activation='linear', input_shape=(2,)),Dense(3, activation='linear'),Dense(1, activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
