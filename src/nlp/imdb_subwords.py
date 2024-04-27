import tensorflow_datasets as tfds

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)

encoder = info.features['text'].encoder
print('Vocabulary size={}'.format(encoder.vocab_size))
print(encoder.subwords)

sample_string = 'Today is a sunny day'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))
print(encoder.subwords[6426])

original_string = encoder.decode(encoded_string)
test_string = encoder.decode([6427, 4869, 9, 4, 2365, 1361, 606])

print('Original string is :{}'.format(original_string))
print('Test string is :{}'.format(test_string))
