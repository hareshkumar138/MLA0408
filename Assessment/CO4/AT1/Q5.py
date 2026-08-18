import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input,LSTM,Dense)
np.random.seed(42)
video_features = np.random.rand(5, 128).astype("float32")
video_features = np.expand_dims(video_features,axis=0)
vocabulary = [
    "a",
    "boy",
    "is",
    "playing",
    "football",
    "girl",
    "running",
    "man",
    "walking"]
vocab_size = len(vocabulary)
encoder_input = Input(shape=(5, 128))
encoder = LSTM(128,return_state=True)
encoder_output, state_h, state_c = encoder(encoder_input)
decoder_input = Input(shape=(None, vocab_size))
decoder_lstm = LSTM(128,return_sequences=True,return_state=True)
decoder_output, _, _ = decoder_lstm(decoder_input,initial_state=[state_h, state_c])
decoder_dense = Dense(vocab_size,activation="softmax")
output = decoder_dense(decoder_output)
model = Model([encoder_input, decoder_input],output)
model.compile(optimizer="adam",loss="categorical_crossentropy")
model.summary()
print("\nVideo caption model created successfully.")
decoder_data = np.random.rand(1, 5, vocab_size).astype("float32")
prediction = model.predict([video_features, decoder_data],verbose=0)
print("\nPrediction shape:",prediction.shape)
print("Video caption generation completed.")
