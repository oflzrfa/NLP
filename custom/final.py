import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow.keras
import string
import random

# Parameters for our model
vocab_size = 10000
max_length = 100
embedding_dim = 50
latent_dim = 32
hidden_units = 64

# Load all files from the "documents" folder
folder_path = "./documents"
file_names = ["text_{}.txt".format(i+1) for i in range(60)]
texts = []
for file_name in file_names:
    with open(os.path.join(folder_path, file_name), encoding="utf-8") as file:
        texts.append(file.read())

# Tokenize and pad the sequences
filters = string.punctuation.replace('.', '')  # keep periods
filters = filters.replace('!', '')  # keep exclamation marks
filters = filters.replace('?', '')  # keep question marks
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", filters=filters)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# One-hot encode sequences for the loss function
one_hot_sequences = tf.one_hot(padded_sequences, depth=vocab_size)

# Define the sampling function
def sampling(args):
    mean, log_var = args
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
    return mean + K.exp(log_var / 2) * epsilon

# Define the Encoder part
inputs = Input(shape=(max_length,))
x = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
x = LSTM(hidden_units, return_sequences=False)(x)  # We only care about the final encoding, not the outputs at each timestep.

# Now we can separate the encoding into the mean and log variance components
mean = Dense(latent_dim)(x)  # Mean encoding
log_var = Dense(latent_dim)(x)  # Log variance encoding
# Use these to get the final encoding (sampling)
z = Lambda(sampling, output_shape=(latent_dim,))([mean, log_var])

# Define the Decoder part
decoder_inputs = Input(shape=(latent_dim,))
x = RepeatVector(max_length)(decoder_inputs)  # Repeat the input for 'max_length' times to have correct input shape for LSTM
x = LSTM(hidden_units, return_sequences=True)(x)  # Returns sequences now.
outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)  # Wrap Dense layer in TimeDistributed to apply it for each time step

# Define the complete VAE model
encoder = Model(inputs, [mean, log_var, z], name="encoder")
decoder = Model(decoder_inputs, outputs, name="decoder")
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name="vae")

# # # # Define the VAE loss
# # # reconstruction_loss = tf.keras.losses.categorical_crossentropy(one_hot_sequences, outputs)
# # # reconstruction_loss *= max_length  # Scale by the sequence length
# # # kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
# # # vae_loss = K.mean(reconstruction_loss + kl_loss)
# # # vae.add_loss(vae_loss)
# # # vae.compile(optimizer=Adam())

# # # # 3. Train the VAE
# # # vae.fit(padded_sequences, one_hot_sequences, epochs=50, batch_size=1)

class VAE(tensorflow.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Convert labels to categorical one-hot encoding
            data_one_hot = tf.one_hot(data, depth=vocab_size)
            reconstruction_loss = tf.reduce_mean(
                tensorflow.keras.losses.categorical_crossentropy(data_one_hot, reconstruction)
            )
            reconstruction_loss *= max_length
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

# create VAE model
vae = VAE(encoder, decoder)
vae.compile(optimizer=Adam())

# 3. Train the VAE
vae.fit(padded_sequences, epochs=200, batch_size=32)


def generate_text(encoder, decoder, start_string):
    # Convert the start string to numbers (vectorizing)
    input_eval = np.array(tokenizer.texts_to_sequences([start_string]))
    # Pad the input sequence to max_length
    input_eval = pad_sequences(input_eval, maxlen=max_length, padding='post')
    #input_eval = tf.expand_dims(input_eval, 0)
    input_eval = np.array(input_eval)


    # Generate initial latent vector through encoding
    mean, log_var, z = encoder.predict(input_eval)

    # Empty array to store our generated words
    generated_sequence = []

    # Generate new sequence data
    sentence_count = 0
    word_count = 0
    while sentence_count < random.randint(50, 100):
        # Predict next word using the decoder
        predictions = decoder.predict(z)
        
        # Calculate probabilities of next words
        probs = tf.nn.softmax(predictions[0, -1, :]).numpy()
        # Sample a word index from the probability distribution
        predicted_id = np.random.choice(range(vocab_size), p=probs)

        # Append the predicted word to the generated sequence
        generated_sequence.append(predicted_id)
        word_count += 1
        
        # Use the current word as the next input to the encoder
        input_eval = np.append(input_eval, [[predicted_id]], axis=1)[:, -max_length:]
        mean, log_var, z = encoder.predict(input_eval)

        word = ''
        if predicted_id == 0:
            word = '<PAD>'
        else:
            word = tokenizer.index_word[predicted_id]

        if '.' in word or '!' in word or '?' in word:
            sentence_count += 1
            print("******Finished sentence ", sentence_count)
        elif word_count > 20:
            word_count = 0
            if '.' not in tokenizer.index_word:
                    tokenizer.word_index['.'] = len(tokenizer.word_index) + 1
                    tokenizer.index_word[tokenizer.word_index['.']] = '.'
            generated_sequence.append(tokenizer.word_index['.'])
            sentence_count += 1


    # Map the generated sequence from word indices to actual words
    generated_text = ' '.join([tokenizer.index_word[idx] if idx in tokenizer.index_word else '<OOV>' for idx in generated_sequence])
    generated_text = start_string + ' ' + generated_text
    return generated_text


# start_string = "Medical racism in the United States"
# print(generate_text(encoder, decoder, start_string))

# Generate and save 10 texts
for i in range(10):
    selected_document = random.choice(texts)
    sentences = selected_document.split('.')
    selected_sentence = random.choice(sentences)
    # Find the index of the first space or delimiter in the selected sentence
    delimiter_index = len(selected_sentence) // 2
    while delimiter_index < len(selected_sentence) and not selected_sentence[delimiter_index].isspace():
        delimiter_index += 1
    # Take the first half of the selected sentence up until the space or delimiter as the start_string
    start_string = selected_sentence[:delimiter_index]
    generated_text = generate_text(encoder, decoder, start_string)
    file_name = f"custom_finished_text_{i+1}.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(generated_text)
    print(f"Text {i+1} generated and saved as {file_name}")

