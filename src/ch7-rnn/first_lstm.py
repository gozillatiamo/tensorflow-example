import string
import json
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

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
        soup = BeautifulSoup(sentence)
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

# print(len(labels))
# print(len(sentences))

training_size = 23000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 20000
max_length = 80
embedding_dim = 64
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
# print(word_index)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    sequences=training_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=padding_type,
)
# print(training_sequences[0])
# print(training_padded[0])

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    sequences=testing_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=padding_type,
)
# print(testing_sequences[0])
# print(testing_padded[0])


# Embededing
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testinig_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        # Multiple LSTM layers, return_sequences=True to allow stacking LSTM layers.
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embedding_dim, return_sequences=True, dropout=0.2)
        ),
        # Add Dropouts to improve the overfitting.
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, dropout=0.2)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# adam = tf.keras.optimizers.Adam(
#     learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False
# )
# Adjust a learning rate, to improve the overfitting.
adam = tf.keras.optimizers.Adam(
    learning_rate=0.000008, beta_1=0.9, beta_2=0.999, amsgrad=False
)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
model.summary()

num_epochs = 30
history = model.fit(
    training_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=2,
)


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
