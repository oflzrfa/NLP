import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

# Add [MASK] token to the text for masked language modeling
text = "[MASK] Today in the world of technology"
inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)

input_ids = inputs['input_ids']
labels = tf.identity(input_ids)

# We mask half of the input tokens
rand = tf.random.uniform(input_ids.shape)
mask_arr = (rand < 0.5) & (input_ids != 101) & (input_ids != 102) # Exclude [CLS] and [SEP] from masking
mask_indices = tf.where(mask_arr)

updates = np.full(mask_indices.shape[0], 103) # 103 is the token ID for [MASK] in BERT
input_ids = tf.tensor_scatter_nd_update(input_ids, mask_indices, updates)

inputs = {"input_ids": input_ids, "labels": labels}

# Train the model
model.compile(optimizer='adam', loss=model.compute_loss)  
model.fit(inputs, epochs=5)

# Now let's generate text
initial_text = "Today in the world of technology"
inputs = tokenizer(initial_text, return_tensors='tf', truncation=True, padding=True)

# Generate a mask for the input
mask = tf.fill(inputs.input_ids.shape, 103)  # Fill the tensor with [MASK] tokens

# Create a mask with the first half tokens set to the original input and the rest set to [MASK]
input_ids = tf.concat([inputs.input_ids[0][:len(inputs.input_ids[0])//2], mask[0][len(inputs.input_ids[0])//2:]], axis=0)
input_ids = tf.expand_dims(input_ids, 0)  # Add batch dimension

# Generate text
predicted_token_ids = model.predict(input_ids)[0].argmax(-1)

predicted_text = tokenizer.decode(predicted_token_ids[0])

print(predicted_text)
