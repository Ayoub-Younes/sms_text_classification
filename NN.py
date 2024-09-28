# Import libraries. You may or may not use all of these.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, StringLookup, CategoryEncoding, IntegerLookup # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # type: ignore

# Import data
current_dir = os.path.dirname(os.path.abspath(__file__))

train_url = 'https://cdn.freecodecamp.org/project-data/sms/train-data.tsv'
train_filename = os.path.join(current_dir, "train-data.tsv")
response_train = requests.get(train_url)

valid_url = 'https://cdn.freecodecamp.org/project-data/sms/train-data.tsv'
valid_filename = os.path.join(current_dir, "valid-data.tsv")
response_valid = requests.get(valid_url)

'''with open(train_filename, "wb") as file:
  file.write(response_train.content)

with open(valid_filename, "wb") as file:
  file.write(response_valid.content)'''
  

train_data = pd.read_csv(train_filename, sep='\t', header = None, names=['label', 'message'])
valid_data = pd.read_csv(valid_filename, sep='\t', header = None, names=['label', 'message'])

print(train_data)
print(valid_data)

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['message'].values)
tokenizer.fit_on_texts(valid_data['message'].values)
sequences_train = tokenizer.texts_to_sequences(train_data['message'].values)
sequences_valid = tokenizer.texts_to_sequences(valid_data['message'].values)

# Padding sequences
maxlen = 50
X_train = pad_sequences(sequences_train, maxlen=maxlen)
X_valid = pad_sequences(sequences_valid, maxlen=maxlen)

# Convert sentiment to binary labels (0 = spam, 1 = ham)
train_data['label'] = train_data['label'].apply(lambda x: 1 if x == 'ham' else 0)
y_train = train_data['label'].values
valid_data['label'] = valid_data['label'].apply(lambda x: 1 if x == 'ham' else 0)
y_valid = valid_data['label'].values


# Define the LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=maxlen),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_valid , y_valid))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_valid , y_valid)
print(f"Test Accuracy: {accuracy}")


# Test the model with new phrases
def predict_sentiment(texts):
    sequence = tokenizer.texts_to_sequences(texts)
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded_sequence)
    return [ 'ham' if pred > 0.5 else 'spam' for pred in prediction.flatten() ]


texts = ["how are you doing today","sale today! to stop texts call 98912460324","i dont want to go. can we try it a different day? available sat","our new mobile video service is live. just install on your phone to start watching.","you have won £1000 cash! call to claim your prize.","i'll bring it tomorrow. don't forget the milk.","wow, is your arm alright. that happened to me one time too"]
print(predict_sentiment(texts))
