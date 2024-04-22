import tensorflow as tf
import tensorflow_datasets as tfds

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))
for item in train_data:
    imdb_sentences.append(str(item['text']))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)

print(tokenizer.word_index)
