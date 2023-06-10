import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback
import glob
import random
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import sys
import os

# Load text data from the files
files = glob.glob("documents/text_*.txt")
text = ""
for file in files:
    with open(file, 'r', encoding="utf-8") as f:
        text += f.read()

# Tokenize the text into words
tokens = word_tokenize(text)
words = sorted(list(set(tokens)))
word_to_int = dict((w, i) for i, w in enumerate(words))
int_to_word = dict((i, w) for i, w in enumerate(words))
vocab_size = len(words)

# Generate input-output pairs
sequence_length = 10
step = 1
sentences = []
next_words = []
for i in range(0, len(tokens) - sequence_length, step):
    sentences.append(tokens[i: i + sequence_length])
    next_words.append(tokens[i + sequence_length])

# Vectorize the data
X = np.zeros((len(sentences), sequence_length), dtype=np.int32)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t] = word_to_int[word]
    y[i, word_to_int[next_words[i]]] = 1

print(X.shape)
print(y.shape)

# Build the model: word-based LSTM
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=sequence_length))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Function to generate text after each epoch
def generate_text(epoch, _):
    print(f'----- Generating text after Epoch: {epoch}')
    
    # Randomly select a seed sentence from the available documents
    seed_sentence = random.choice(sentences)
    
    generated_text = ' '.join(seed_sentence)
    
    print(f'----- Generating with seed: "{generated_text}"')
    sys.stdout.write(generated_text)
    
    while len(generated_text.split()) < 200:  # Adjust the desired length as needed
        x_pred = np.zeros((1, sequence_length), dtype=np.int32)
        input_sequence = generated_text.split()[-sequence_length:]
        for t, word in enumerate(input_sequence):
            x_pred[0, t] = word_to_int[word]
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_word = int_to_word[next_index]
        
        generated_text += ' ' + next_word
        
        sys.stdout.write(' ' + next_word)
        sys.stdout.flush()
    
    print("\nGenerated text:\n", generated_text)

print_callback = LambdaCallback(on_epoch_end=generate_text)

model.fit(X, y, batch_size=128, epochs=5, callbacks=[print_callback])
