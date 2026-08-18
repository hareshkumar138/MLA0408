import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input,Embedding,Bidirectional,LSTM,Dense)
vocab_size = 20
sequence_length = 5
encoder_data = np.array([
    [1, 2, 3, 0, 0],
    [1, 4, 3, 0, 0],
    [1, 5, 6, 3, 0],
    [1, 2, 7, 3, 0]])
decoder_data = np.array([
    [1, 8, 9, 3, 0],
    [1, 10, 9, 3, 0],
    [1, 11, 12, 9, 3],
    [1, 8, 13, 9, 0]])
encoder_input = Input(shape=(sequence_length,))
embedding = Embedding(input_dim=vocab_size,output_dim=64)(encoder_input)
encoder_output = Bidirectional(LSTM(64))(embedding)
decoder_input = Input(shape=(sequence_length,))
decoder_embedding = Embedding(input_dim=vocab_size,output_dim=64)(decoder_input)
context = tf.keras.layers.RepeatVector(sequence_length)(encoder_output)
decoder_lstm = LSTM(128,return_sequences=True)(context)
decoder_output = Dense(vocab_size,activation="softmax")(decoder_lstm)
model = Model(encoder_input,decoder_output)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()
print("\nBidirectional Encoder-Decoder model created successfully.")
test_sentence = np.array([[1, 2, 3, 0, 0]])
prediction = model.predict(test_sentence,verbose=0)
print("Translation prediction shape:")
print(prediction.shape)
