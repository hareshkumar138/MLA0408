import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense,Embedding
np.random.seed(42)
frames=np.random.rand(20,64,64,3).astype("float32")
frames=preprocess_input(frames)
cnn=MobileNetV2(weights="imagenet",include_top=False,pooling="avg",input_shape=(64,64,3))
cnn.trainable=False
features=cnn.predict(frames,verbose=0)
features=features.reshape(4,5,1280)
captions=np.array([[1,5,10,15,20],[1,6,11,16,21],[1,7,12,17,22],[1,8,13,18,23]])
targets=np.expand_dims(captions,-1)
encoder_inputs=Input(shape=(5,1280))
_,state_h,state_c=LSTM(64,return_state=True)(encoder_inputs)
decoder_inputs=Input(shape=(5,))
x=Embedding(50,64)(decoder_inputs)
x=LSTM(64,return_sequences=True)(x,initial_state=[state_h,state_c])
output=Dense(50,activation="softmax")(x)
model=Model([encoder_inputs,decoder_inputs],output)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit([features,captions],targets,epochs=2,batch_size=2,verbose=1)
prediction=model.predict([features[:1],captions[:1]],verbose=0)
words=np.argmax(prediction[0],axis=-1)
print("Video Feature Shape:",features.shape)
print("Prediction Shape:",prediction.shape)
print("Predicted Word IDs:",words)
print("Video Captioning Model Executed Successfully.")
