import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pa
import newspaper
from gensim.models import Word2Vec
from transformers import TFBertForSequenceClassification, BertTokenizer, TFBertForMaskedLM
import random
import tensorflow as tf

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForMaskedLM.from_pretrained(model_name)

text = "[MASK] Today in the world of technology"
inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
input_ids = inputs.input_ids
labels = tf.identity(input_ids)
rand = tf.random.uniform(input_ids.shape)
mask_arr = (rand < 0.5) & (input_ids != 101) & (input_ids != 102) # Exclude [CLS] and [SEP] from masking

mask_indices = tf.where(mask_arr)
updates = np.full(mask_indices.shape[0], 103) # 103 is the token ID for [MASK] in BERT

input_ids = tf.tensor_scatter_nd_update(input_ids, mask_indices, updates)


inputs = {"input_ids": input_ids, "labels": labels}

model.compile(optimizer='adam', loss=model.compute_loss)
model.fit(inputs['input_ids'], epochs=5)

# Generate text with the transformer model
generated_text_samples = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=random.randint(500, 600),  # The length of the generated text
    temperature=0.7,
    num_return_sequences=10  # The number of different text samples you want to generate
)

# Decode the generated tokens to text
for i, generated_text in enumerate(generated_text_samples):
    print(f"=== GENERATED TEXT {i+1} ===")
    print(tokenizer.decode(generated_text, skip_special_tokens=True))
