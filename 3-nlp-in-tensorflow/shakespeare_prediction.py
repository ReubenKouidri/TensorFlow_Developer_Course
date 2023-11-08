from typing import Any, Optional
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

SONNETS_FILE = 'shakespeare/sonnets.txt'


def n_gram_seqs(corpus: list[str], tokenizer: Tokenizer) -> list[list[int]]:
    """Generates a list of n-gram sequences for a given corpus

    Args:
        corpus (list of string): lines of texts to generate n-grams for
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary

    Returns:
        n_grams (list of lists of int): the n-gram sequences for each line in the corpus
    """
    n_grams = []
    for line in corpus:
        sequence = tokenizer.texts_to_sequences([line])[0]
        n_grams.extend(sequence[: i + 1] for i in range(1, len(sequence)))
    return n_grams


def pad_seqs(input_sequences: list, maxlen: int) -> np.ndarray:
    """
    Pads tokenized sequences to the same length

    Args:
        input_sequences (list of int): tokenized sequences to pad
        maxlen (int): maximum length of the token sequences

    Returns:
        padded_sequences (array of int): tokenized sequences padded to the same length
    """
    padded_sequences = np.array(pad_sequences(input_sequences, padding='pre', maxlen=maxlen))
    return padded_sequences


def read_data(sonnets_file: str) -> list:
    """
    Reads the sonnets_file and returns a list of lines

    Args:
        sonnets_file (str): path to the sonnets.txt file

    Returns:
        corpus (list of string): list of lines of the sonnets.txt file
    """
    with open(sonnets_file, 'r') as f:
        data = f.read()
    corpus = data.lower().split("\n")
    return corpus


def fit_tokenizer(corpus: list, tokenizer: Tokenizer) -> Tokenizer:
    """
    Fits a tokenizer on the given corpus

    Args:
        corpus (list of string): corpus to fit the tokenizer on

    Returns:
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
    tokenizer.fit_on_texts(corpus)
    return tokenizer


def features_and_labels(input_sequences: np.ndarray, total_words: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates features and labels from n-grams

    Args:
        input_sequences (list of lists of int): sequences of n-grams from which to generate features and labels
        total_words (int): vocabulary size

    Returns:
        features (type(input_sequences)), one_hot_labels (2D array): arrays of features and one-hot encoded labels
    """
    features = input_sequences[:, :-1]  # all rows up to -1 column
    labels = input_sequences[:, -1]  # all rows, last element

    one_hot_labels = tf.keras.utils.to_categorical(labels, total_words)
    return features, one_hot_labels


def create_model(total_words: int, max_sequence_len: int) -> tf.keras.Model:
    """Creates a text generator model

    Args:
        total_words (int): size of the vocabulary for the Embedding layer input
        max_sequence_len (int): length of the input sequences

    Returns:
        model (tf.keras Model): the text generator model
    """
    lstm_units = 32
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True))),
    model.add(Bidirectional(LSTM(lstm_units))),
    model.add(Dense(512, activation='relu')),
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def plot_history(history: Any) -> None:
    """Plots training history

    Args:
        history (object): the training history
    """
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()


def make_prediction(seed_text: str, tokenizer: Tokenizer, num_words: Optional[int] = 10) -> str:
    """
    Generates a sequences of num_words predictions based on a given seed text.

    Args:
        seed_text (str): The initial text from which the prediction is generated.
        tokenizer: The tokenizer used to convert the text into sequences.
        num_words (int, optional): The number of words to generate in the prediction. Defaults to 10.

    Returns:
        string: The generated text.
    """
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]  # Convert the text into sequences
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')  # Pad
        predicted = model.predict(token_list, verbose=0)  # Get the probabilities of predicting a word
        predicted = np.argmax(predicted, axis=-1).item()  # Choose the next word based on the maximum probability
        output_word = tokenizer.index_word[predicted]  # Get the actual word from the word index
        seed_text += " " + output_word  # Append to the current text
    return seed_text


if __name__ == "__main__":
    corpus = read_data(SONNETS_FILE)

    tokenizer = Tokenizer()
    tokenizer = fit_tokenizer(corpus, tokenizer)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = n_gram_seqs(corpus, tokenizer)
    max_sequence_len = max([len(x) for x in input_sequences])

    padded_inputs = pad_seqs(input_sequences, max_sequence_len)
    features, labels = features_and_labels(padded_inputs, total_words)

    model = create_model(total_words, max_sequence_len)
    history = model.fit(features, labels, epochs=50, verbose=1)
    plot_history(history)

    make_prediction(seed_text="Help me Obi Wan Kenobi, you're my only hope", tokenizer=tokenizer, num_words=10)
