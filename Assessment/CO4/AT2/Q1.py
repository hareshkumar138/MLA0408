import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
np.random.seed(42)
X=np.random.rand(40,224,224,3).astype("float32")
y=np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
X_train,X_test=X[:32],X[32:]
y_train,y_test=y[:32],y[32:]
base=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
base.trainable=False
x=GlobalAveragePooling2D()(base.output)
x=Dense(128,activation="relu")(x)
x=Dropout(0.5)(x)
output=Dense(1,activation="sigmoid")(x)
model=Model(base.input,output)
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=2,batch_size=4,validation_split=0.2)
loss,accuracy=model.evaluate(X_test,y_test,verbose=0)
print("Test Accuracy:",round(accuracy*100,2),"%")
print("Prediction:", "Malignant" if model.predict(X_test[:1],verbose=0)[0][0]>=0.5 else "Benign")
