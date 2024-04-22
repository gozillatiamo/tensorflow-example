import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'is it sunny today?',
]

# Add oov token for tokenizer
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
print("===== Common texts_to_sequences START =====")
print(word_index)
print(sequences)
print("===== Common texts_to_sequences END =====")


# OOV
print("===== OOV texts_to_sequences START =====")
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)
print("===== OOV texts_to_sequences END =====")

# Padding
print("===== Padding texts_to_sequences START =====")
sentences.append('I really enjoyed walking in the snow today')
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
print(padded)

print("===== Padding texts_to_sequences END =====")
