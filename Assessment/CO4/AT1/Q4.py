import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
prices = np.array([
    100, 102, 104, 103, 106,
    108, 110, 109, 112, 115,
    117, 116, 120, 122, 125,
    124, 128, 130, 133, 135
], dtype=float)
minimum = prices.min()
maximum = prices.max()
prices = (prices - minimum) / (maximum - minimum)
X = []
y = []
sequence_length = 3
for i in range(len(prices) - sequence_length):
    X.append(prices[i:i + sequence_length])
    y.append(prices[i + sequence_length])
X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0],X.shape[1],1)
model = Sequential([LSTM(50,input_shape=(3, 1)),Dense(1)])
model.compile(optimizer="adam",loss="mse")
model.fit(X,y,epochs=20,batch_size=4,verbose=0)
print("LSTM training completed.")
last_three = prices[-3:]
test_input = last_three.reshape(1, 3, 1)
prediction = model.predict(test_input,verbose=0)
predicted_price = (prediction[0][0] *(maximum - minimum)+ minimum)
print("Predicted next closing price:",round(float(predicted_price), 2))
