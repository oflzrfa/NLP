import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Read the text data from the files
documents_folder = "documents"
file_names = [f"text_{i}.txt" for i in range(1, 61)]
texts = []
for file_name in file_names:
    with open(os.path.join(documents_folder, file_name), "r", encoding="utf-8") as f:
        text = f.read()
        texts.append(text)

# Preprocess the text data
tokenizer = Tokenizer(lower=True, oov_token="<unk>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to a fixed length
max_sequence_length = 100  # Adjust as needed
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")

# Convert the padded sequences into input and target sequences
input_sequences = padded_sequences[:, :-1]
target_sequences = padded_sequences[:, 1:]

# Convert the target sequences to one-hot encoded vectors
num_words = len(tokenizer.word_index) + 1
target_sequences_one_hot = to_categorical(target_sequences, num_classes=num_words)

# Define the VAE model architecture
latent_dim = 32

# Encoder
encoder_inputs = Input(shape=(max_sequence_length-1,num_words))
encoder_lstm = LSTM(64, return_sequences=True)(encoder_inputs)
z_mean = Dense(latent_dim)(encoder_lstm)
z_log_var = Dense(latent_dim)(encoder_lstm)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = Input(shape=(latent_dim,num_words))
decoder_lstm = LSTM(64, return_sequences=True)(decoder_inputs)
decoder_outputs = Dense(num_words, activation='softmax')(decoder_lstm)

# VAE model
vae = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

# Reshape target sequences
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length-1, padding='post')
decoder_outputs = decoder_outputs[:, :max_sequence_length-1, :]

# Define the loss function
reconstruction_loss = mse(target_sequences, decoder_outputs)
kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)

# Compile and train the VAE model
vae.compile(optimizer='adam')
vae.fit([input_sequences, np.zeros_like(input_sequences)], batch_size=128, epochs=10)

# Generate text using the trained VAE with beam search
beam_width = 5
max_length = 100

def generate_text_beam_search(vae, beam_width, max_length, start_seq):
    start_seq = pad_sequences([start_seq], maxlen=max_sequence_length-1, padding='post', truncating='post')
    hidden_state = vae.encoder.predict(start_seq)
    initial_state = np.zeros((beam_width, latent_dim))
    initial_state[0] = hidden_state
    
    current_sequences = [[start_seq, 0.0, initial_state]]
    completed_sequences = []

    # Implement beam search
    while len(completed_sequences) < beam_width:
        new_sequences = []

        for seq, score, state in current_sequences:
            if len(seq[0]) >= max_length-1:
                completed_sequences.append((seq, score))
                continue

            next_word_probs = vae.decoder.predict([state, seq])[0][-1]
            top_k_indices = np.argsort(next_word_probs)[-beam_width:]

            for idx in top_k_indices:
                next_seq = np.append(seq, [[idx]], axis=1)
                next_score = score + np.log(next_word_probs[idx])
                next_state = state.copy()
                next_state[0] = hidden_state

                new_sequences.append([next_seq, next_score, next_state])

        new_sequences = sorted(new_sequences, key=lambda x: x[1], reverse=True)[:beam_width]
        current_sequences = []

        for seq, score, state in new_sequences:
            if seq[0][-1] == 0:
                completed_sequences.append((seq, score))
            else:
                current_sequences.append((seq, score, state))

    best_seq, best_score = completed_sequences[0]
    generated_text = ' '.join([tokenizer.index_word[word_idx] for word_idx in best_seq[0]])
    return generated_text

# Generate text using beam search
start_sequence = [tokenizer.word_index['<start>']]  # Customize the starting sequence as needed
generated_text_beam_search = generate_text_beam_search(vae, beam_width, max_length, start_sequence)
