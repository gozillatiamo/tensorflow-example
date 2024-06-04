import json 
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
from collections import OrderedDict

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

table = str.maketrans('', '', string.punctuation)
with open("content/sarcasm.json", 'r') as f:
    datastore = json.load(f)
    sentences = []
    labels = []
    urls = []
    for item in datastore:
        sentence = item['headline'].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence=""
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

def plot_sentences(sentences):
    xs=[]
    ys=[]
    current_item=1

    for item in sentences:
        xs.append(current_item)
        current_item=current_item+1

        ys.append(len(item))

        newys = sorted(ys)

    plt.plot(xs, newys)
    plt.show()


# plot_sentences(training_sentences)

# vocab_size = 10000 
vocab_size = 2000 
# max_length = 100
max_length = 80
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
# print(word_index)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    sequences=training_sequences, 
    maxlen=max_length,
    padding= padding_type,
    truncating=padding_type  
)
# print(training_sequences[0])
# print(training_padded[0])

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    sequences=testing_sequences, 
    maxlen=max_length,
    padding= padding_type,
    truncating=padding_type  
)
# print(testing_sequences[0])
# print(testing_padded[0])


# Embededing
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testinig_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    # Embedding each word to 16 dimention array, Parameters: 10000*16 = 160000
    # tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.Embedding(vocab_size, 7),
    # Find parameter for Embedding to create a 16 dimention vector, Parameters: 0
    tf.keras.layers.GlobalAveragePooling1D(),
    # Weight: 24, Bias: 1, Parameters: (24*16)+16 = 408
    # tf.keras.layers.Dense(24, activation='relu'),
    # tf.keras.layers.Dense(8, activation='relu'),
    # Regularization L2
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    # Use Dropout technique, the result not good when the neuron units is uncomplex
    # tf.keras.layers.Dropout(.25),

    # The last layer, Parameters: (1*24)+1 = 25
    tf.keras.layers.Dense(1, activation='sigmoid')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# num_epochs = 30
num_epochs = 100
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# plot graph accuracy
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

def plot_word_frequency(ordered_words):
    xs=[]
    ys=[]
    curr_x = 1
    for item in ordered_words:
        xs.append(curr_x)
        curr_x=curr_x+1
        ys.append(ordered_words[item])

    plt.plot(xs,ys)
    plt.axis([300, 10000, 0, 100])
    plt.show()

# Predict data
sentences = ["granny starting to fear spiders in the garden might be real", "game of thrones season final showing this sunday night", "TensorFlow book will be a best seller"]
sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# print(padded)
print(model.predict(padded))

# wc=tokenizer.word_counts
# ordered_words = (OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True)))
# print(ordered_words)
# plot_word_frequency(ordered_words)
