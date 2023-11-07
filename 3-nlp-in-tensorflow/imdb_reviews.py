import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.optimizers.legacy as legacy
import io

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LEN = 128
TRUNC_TYPE = "post"
OOV_TOKEN = "<OOV>"


def train_test_data():
    imdb = tfds.load("imdb_reviews", with_info=False, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']
    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    return training_sentences, training_labels_final, testing_sentences, testing_labels_final


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=legacy.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model


def download_visualisation_data(model, tokenizer):
    embedding_layer = model.layers[0]
    embedding_weights = embedding_layer.get_weights()[0]
    print(embedding_weights.shape)
    reverse_word_index = tokenizer.index_word

    with io.open('meta.tsv', 'w', encoding='utf-8') as f, io.open('vecs.tsv', 'w', encoding='utf-8') as o:
        for word_num in range(1, VOCAB_SIZE):
            word_name = reverse_word_index[word_num]
            word_embedding = embedding_weights[word_num]
            f.write(word_name + "\n")
            o.write('\t'.join([str(x) for x in word_embedding]) + "\n")


if __name__ == "__main__":
    train_sentences, train_labels, test_sentences, test_labels = train_test_data()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_sentences)  # tokenizer fit on training set, so test set may contain more OOVs
    # word_index = tokenizer.word_index
    # print(word_index)
    sequences = tokenizer.texts_to_sequences(train_sentences)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE)

    testing_sequences = tokenizer.texts_to_sequences(test_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE)

    model = get_model()
    print(model.summary())
    model.fit(padded, train_labels, epochs=10, validation_data=(testing_padded, test_labels))

    # download_visualisation_data(model, tokenizer)
