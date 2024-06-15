import string
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Stop words config
stopwords = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "hed",
    "hes",
    "her",
    "here",
    "heres",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "hows",
    "i",
    "id",
    "ill",
    "im",
    "ive",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "lets",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "nor",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "shed",
    "shell",
    "shes",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "thats",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "theres",
    "these",
    "they",
    "theyd",
    "theyll",
    "theyre",
    "theyve",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "wed",
    "well",
    "were",
    "weve",
    "were",
    "what",
    "whats",
    "when",
    "whens",
    "where",
    "wheres",
    "which",
    "while",
    "who",
    "whos",
    "whom",
    "why",
    "whys",
    "with",
    "would",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
]

table = str.maketrans("", "", string.punctuation)
# Config hyper parametes
vocab_size = 13200
embedding_dim = 25
max_length = 50
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 23000

# Config the words dictionary
glove_embeddings = dict()
f = open("content/glove.twitter.27B.25d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    glove_embeddings[word] = coefs
f.close()

# print(glove_embeddings["OOV"])

# Preparing the training data and testing data
with open("content/sarcasm.json", "r") as f:
    datastore = json.load(f)


sentences = []
labels = []
urls = []
for item in datastore:
    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence, features="html.parser")
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Tokenize the sarcasm data to the word index
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
# print(word_index)
# print(len(word_index))
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Create a embedding matrix for corpus
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        # print(f"Vector values of {word} is {embedding_vector}")
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embedding_dim, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False
)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
model.summary()
history = model.fit(
    training_padded,
    training_labels,
    epochs=150,
    validation_data=(testing_padded, testing_labels),
    verbose=2,
)


# Finding a word cumulative
def plot_word_cumulative(tokenizer, glove_embeddings):
    xs = []
    ys = []
    cumulative_x = []
    cumulative_y = []
    total_y = 0
    for word, index in tokenizer.word_index.items():
        xs.append(index)
        cumulative_x.append(index)
        if glove_embeddings.get(word) is not None:
            total_y = total_y + 1
            ys.append(1)
        else:
            ys.append(0)
        cumulative_y.append(total_y / index)
        # print(f"{word}:{index}")
        # print(f"Total_y: {total_y}")
    # fig, ax = plt.subplots(figsize=(12, 2))
    # ax.spines["top"].set_visible(False)

    # plt.margins(x=0, y=None, tight=True)
    # plt.fill(ys)
    # print(cumulative_x)
    # print(cumulative_y)
    plt.plot(cumulative_x, cumulative_y)
    # plt.axis([0, 25000, 0.915, 0.985])
    # Zoom axis
    plt.axis([13000, 14000, 0.97, 0.98])
    plt.show()


# plot_word_cumulative(tokenizer, glove_embeddings)


# plot graph accuracy
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

test_sentences = [
    "It Was, For, Uh, Medical Reasons, Says Doctor To Boris Johnson, Explaining Why They Had To Give Him Haircut",
    "It's a beautiful sunny day",
    "I lived in Ireland, so in High School they made me learn to speak and write in Gaelic",
    "Census Foot Soldiers Swarm Neighborhoods, Kick Down Doors To Tally Household Sizes",
]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

print(model.predict(test_padded))
