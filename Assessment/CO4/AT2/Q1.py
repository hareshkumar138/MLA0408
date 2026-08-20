import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Model
X=np.random.rand(8,64,64,3).astype("float32")
y=np.array([0,1,0,1,0,1,0,1])
base=MobileNetV2(weights="imagenet",include_top=False,input_shape=(64,64,3))
base.trainable=False
x=GlobalAveragePooling2D()(base.output)
output=Dense(1,activation="sigmoid")(x)
model=Model(base.input,output)
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X,y,epochs=1,batch_size=4,verbose=1)
loss,accuracy=model.evaluate(X,y,verbose=0)
print("Test Accuracy:",round(accuracy*100,2),"%")
print("Prediction:","Malignant" if model.predict(X[:1],verbose=0)[0][0]>=0.5 else "Benign")
