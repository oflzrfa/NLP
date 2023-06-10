import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import LambdaCallback
import nltk
import sys
import random
import regex as re

#nltk.download('reuters')       # only if you didnt
from nltk.corpus import reuters

# Load Reuters corpus
text = ' '.join(reuters.raw(fileid) for fileid in reuters.fileids())

# Create a mapping from unique characters to integers, and a reverse mapping
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Cut the corpus into chunks of 100 characters, stepping 3 characters at a time
sequence_length = 100
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - sequence_length, step):
    sentences.append(text[i: i + sequence_length])
    next_chars.append(text[i + sequence_length])

# Vectorization
X = np.zeros((len(sentences), sequence_length, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

print(X.size)
print(y.size)
print(X.shape)
print(y.shape)

# Build the model: a single LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Function to generate text after each epoch
def generate_text(epoch, _):
    print(f'----- Generating text after Epoch: {epoch}')
    generated = ''
    # start_index = np.random.randint(0, len(text) - sequence_length - 1)
    
    # sentence = text[start_index: start_index + sequence_length]
    # generated += sentence

    found = False
    sentence = ''
    while not found:
        random_article = np.random.choice(reuters.fileids())  # Randomly select a news article
        article_text = reuters.raw(random_article)
        start_sentences = re.findall(r'[A-Z][^.!?]*[.!?]', article_text)
        if (len(start_sentences)): found = True
        if found: sentence = start_sentences[0]

    generated = sentence
    
    print(f'----- Generating with seed: "{sentence}"')
    sys.stdout.write(generated)
    generated_text = generated

    num_sentences = 1
    sentence_enders = ['.', '?', '!']

    while num_sentences < random.randint(50,100):
        x_pred = np.zeros((1, sequence_length, len(chars)))
        for t, char in enumerate(sentence):
            if t < sequence_length:
                x_pred[0, t, char_to_int[char]] = 1.
            
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = int_to_char[next_index]
        
        sentence = sentence[1:] + next_char
        
        # sys.stdout.write(next_char)
        # sys.stdout.flush()
        generated_text += next_char

        if next_char in sentence_enders:
            num_sentences += 1
        elif next_char == ' ':
            if len(generated_text) >= sequence_length-1 or len(generated_text) > 1 and generated_text[-2] in sentence_enders: 
                generated_text = generated_text[:-1] + '.'
                num_sentences += 1

    # Save generated text to file
    filename = f"generated_article_{epoch}.txt"
    with open(filename, 'w') as file:
        file.write(generated_text)

    print(f"\nGenerated text saved to {filename}\n")

print_callback = LambdaCallback(on_epoch_end=generate_text)

model.fit(X, y, batch_size=128, epochs=1, callbacks=[print_callback])

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
model.save_weights('my_model_weights.h5')

