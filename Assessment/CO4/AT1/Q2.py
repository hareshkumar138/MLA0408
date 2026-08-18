import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model
np.random.seed(42)
image = np.random.rand(1, 224, 224, 3).astype("float32")
num_classes = 5
base_model = DenseNet121(weights=None,include_top=False,input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
inputs = base_model.input
x = base_model.output
x = Conv2D(128,(3, 3),padding="same",activation="relu")(x)
x = UpSampling2D(size=(8, 8))(x)
outputs = Conv2D(num_classes,(1, 1),activation="softmax",padding="same")(x)
model = Model(inputs,outputs)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
prediction = model.predict(image,verbose=0)
segmentation = np.argmax(prediction[0],axis=-1)
print("Input Image Shape:", image.shape)
print("Segmentation Output Shape:", segmentation.shape)
print("\nPixel-wise segmentation completed.")
print("Example pixel classes:")
print(segmentation[100:105, 100:105])
