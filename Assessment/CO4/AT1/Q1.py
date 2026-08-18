import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
np.random.seed(42)
X = np.random.rand(20, 224, 224, 3).astype("float32")
y = np.array([
    0, 1, 0, 1, 0,
    1, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
    1, 0, 1, 0, 1
])
base_model = VGG16(weights=None,include_top=False,input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation="relu")(x)
output = Dense(2, activation="softmax")(x)
model = Model(inputs=base_model.input,outputs=output)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(X,y,epochs=2,batch_size=4,verbose=1)
test_image = np.random.rand(1, 224, 224, 3).astype("float32")
prediction = model.predict(test_image,verbose=0)
class_names = ["Healthy", "Disease"]
result = np.argmax(prediction)
print("\nPlant Disease Prediction:")
print("Result:", class_names[result])
print("Confidence:", round(float(prediction[0][result]) * 100, 2), "%")
