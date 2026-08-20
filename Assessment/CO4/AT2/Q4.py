import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
np.random.seed(42)
prices=(100+np.cumsum(np.random.normal(0,1,200))).reshape(-1,1)
scaler=MinMaxScaler()
data=scaler.fit_transform(prices)
X=[]
y=[]
for i in range(20,len(data)):
    X.append(data[i-20:i])
    y.append(data[i])
X=np.array(X)
y=np.array(y)
model=Sequential([LSTM(64,return_sequences=True,input_shape=(X.shape[1],X.shape[2])),Dropout(0.2),LSTM(32),Dropout(0.2),Dense(1)])
model.compile(optimizer="adam",loss="mean_squared_error")
model.fit(X,y,epochs=5,batch_size=16,validation_split=0.2)
predicted=scaler.inverse_transform(model.predict(X,verbose=0))
print("Actual Prices:",prices[-5:].flatten())
print("Predicted Prices:",predicted[-5:].flatten())
plt.plot(prices,label="Actual")
plt.plot(range(20,len(prices)),predicted,label="Predicted")
plt.title("LSTM Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
