import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input,Conv2D,UpSampling2D
from tensorflow.keras.models import Model
np.random.seed(42)
X=np.random.rand(20,128,128,3).astype("float32")
Y=np.random.randint(0,4,(20,128,128,1)).astype("int32")
inputs=Input(shape=(128,128,3))
base=DenseNet121(weights="imagenet",include_top=False,input_tensor=inputs)
base.trainable=False
x=Conv2D(256,3,padding="same",activation="relu")(base.output)
x=UpSampling2D((2,2))(x)
x=Conv2D(128,3,padding="same",activation="relu")(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(64,3,padding="same",activation="relu")(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(32,3,padding="same",activation="relu")(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(16,3,padding="same",activation="relu")(x)
x=UpSampling2D((2,2))(x)
output=Conv2D(4,1,activation="softmax")(x)
model=Model(inputs,output)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(X,Y,epochs=2,batch_size=2)
prediction=model.predict(X[:1],verbose=0)
segmentation=np.argmax(prediction,axis=-1)
print("Segmentation Shape:",segmentation.shape)
print("0=Road  1=Pedestrian  2=Vehicle  3=Traffic Sign")
