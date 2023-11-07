from __future__ import annotations
import os
import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.optimizers.legacy as legacy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from typing import Any


BASE_DIR = "./bbc"
BUSINESS_DIR = os.path.join(BASE_DIR, "business")
ENTERTAINMENT_DIR = os.path.join(BASE_DIR, "entertainment")
POLITICS_DIR = os.path.join(BASE_DIR, "politics")
SPORT_DIR = os.path.join(BASE_DIR, "sport")
TECH_DIR = os.path.join(BASE_DIR, "tech")
SPLIT = 0.8

OOV_TOK = "<OOV>"
NUM_WORDS = 10000
MAXLEN = 120
EMBEDDING_DIM = 16

random.seed(123)
tf.random.set_seed(123)


def strip_blank_lines(filename: str | os.PathLike) -> str:
    """
    Reads the contents of a file and returns a list of non-blank lines.

    Args:
        filename (str): The name of the file to be read.

    Returns:
        str: The contents of the file as a string without any blank lines.
    """
    with open(filename, 'r') as f:
        non_blank_lines = "".join([line.strip() for line in f if line.strip()][1:])
    return non_blank_lines


def get_data(directory: str | os.PathLike) -> list[tuple[str, str]]:
    """
    Given a directory path, this function retrieves the text files from the directory and its subdirectories,
    processes the text data, and returns a list of tuples containing the processed text and the corresponding label.

    Parameters:
    - directory (str): The path to the directory containing the text files.

    Returns:
    - data (list): A list of tuples, where each tuple contains the processed text (str) and the label (str) of the text file.
    """
    data = []
    label = os.path.basename(directory)
    files = sorted(glob.glob(os.path.join(directory, '*.txt')), reverse=False)

    for file in files:
        text = strip_blank_lines(file)
        text = remove_stopwords(text)
        data.append((text, label))
    return data


def remove_stopwords(sentence: str) -> str:
    """
    Removes stopwords from a given sentence.

    Args:
        sentence (str): The sentence from which stopwords will be removed.

    Returns:
        str: The sentence with stopwords removed.
    """
    stopwords = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
                 "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it",
                 "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once",
                 "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
                 "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
                 "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where",
                 "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you",
                 "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"}

    words = sentence.lower().split()
    new_sentence = [word for word in words if word not in stopwords]
    return " ".join(new_sentence)


def train_val_split(data: list, training_split: float = 0.8) -> tuple[list, list]:
    """
    Split the given data into training and validation sets.

    Args:
        data (list): The input data to be split.
        training_split (float, optional): The fraction of the data to be used for training. Defaults to 0.8.

    Returns:
        tuple: A tuple containing the training set and the validation set.
    """
    train_size = int(len(data) * training_split)
    return data[:train_size], data[train_size:]


def seq_and_pad(sentences: list, tokenizer: Tokenizer, padding: str, maxlen: int) -> np.ndarray:
    """
    Generates sequences from a list of sentences using a tokenizer and pads them to a maximum length.

    Args:
        sentences (list): A list of sentences.
        tokenizer (Tokenizer): An instance of the Tokenizer class.
        padding (str): The type of padding to apply to the sequences.
        maxlen (int): The maximum length of the sequences.

    Returns:
        np.ndarray: An array of padded sequences.
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding=padding, maxlen=maxlen)
    return padded_sequences


def fit_tokenizer(train_sentences: list, num_words: int = NUM_WORDS, oov_tok: str = OOV_TOK) -> Tokenizer:
    """
    Fits a tokenizer on the given training sentences.

    Args:
        train_sentences (list): A list of training sentences.
        num_words (int, optional): The maximum number of words to keep. Defaults to NUM_WORDS.
        oov_tok (str, optional): The token to use for out-of-vocabulary words. Defaults to OOV_TOK.

    Returns:
        Tokenizer: The fitted tokenizer.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    return tokenizer


def tokenize_labels(all_labels: list, split_labels: list) -> np.ndarray:
    """
    Tokenizes the labels

    Args:
        all_labels (list of string): labels to generate the word-index from
        split_labels (list of string): labels to tokenize

    Returns:
        label_seq_np (array of int): tokenized labels
    """
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(all_labels)  # Fit the tokenizer on all the labels
    label_seq = label_tokenizer.texts_to_sequences(split_labels)  # Convert split labels to sequences
    label_seq_np = np.array(label_seq) - 1  # Convert sequences to a numpy array and subtract 1 (start from 0)
    return label_seq_np


def create_model(num_words: int, embedding_dim: int, maxlen: int) -> tf.keras.Model:
    """Creates a text classifier model
    Args:
        num_words (int): size of the vocabulary for the Embedding layer input
        embedding_dim (int): dimensionality of the Embedding layer output
        maxlen (int): length of the input sequences

    Returns:
        model (tf.keras Model): the text classifier model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=legacy.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model


def plot_graphs(history: Any, metric: str) -> None:
    """
    Plot graphs of the given history and metric.

    Parameters:
        history (object): The history object containing the training metrics.
        metric (str): The metric to plot on the graph.

    Returns:
        None
    """
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()


if __name__ == '__main__':
    train_data = []
    validation_data = []
    for directory in os.listdir(BASE_DIR):
        abs_dir = os.path.join(BASE_DIR, directory)
        if os.path.isdir(abs_dir):
            data = get_data(abs_dir)
            train, val = train_val_split(data, SPLIT)
            train_data.extend(train)
            validation_data.extend(val)

    random.shuffle(train_data)
    random.shuffle(validation_data)

    train_sentences = [sentence[0] for sentence in train_data]
    train_labels = [sentence[1] for sentence in train_data]

    validation_sentences = [sentence[0] for sentence in validation_data]
    validation_labels = [sentence[1] for sentence in validation_data]

    all_labels = train_labels + validation_labels

    tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOK)
    train_padded_sequences = seq_and_pad(train_sentences, tokenizer, padding='post', maxlen=MAXLEN)
    val_padded_sequences = seq_and_pad(validation_sentences, tokenizer, padding='post', maxlen=MAXLEN)

    train_label_seq = tokenize_labels(all_labels, train_labels)
    val_label_seq = tokenize_labels(all_labels, validation_labels)

    model = create_model(NUM_WORDS, EMBEDDING_DIM, MAXLEN)
    history = model.fit(train_padded_sequences,
                        train_label_seq,
                        epochs=30,
                        validation_data=(val_padded_sequences, val_label_seq))

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
