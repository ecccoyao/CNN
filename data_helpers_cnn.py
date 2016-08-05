import numpy as np
import pickle


def pad_sentences(sentences,seq_size):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    return [sentence + ["."] * (seq_size - len(sentence)) for sentence in sentences]


def build_input_data(sentences, labels, vocabulary_index):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary_index.get("unknown") if vocabulary_index.get(word) is None \
                   else vocabulary_index.get(word) for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def create_case_data(cases,seq_size,vocabulary_index):
    x_text = [s.split(" ")[:seq_size] for s in cases]
    x_text = pad_sentences(x_text,seq_size)
    x = np.array([[vocabulary_index.get("unknown") if vocabulary_index.get(word) is None \
                       else vocabulary_index.get(word) for word in sentence] for sentence in x_text])
    return x


def load_obj(file, name):
    """
    load the model from the pickle
    :param name: file Name
    :return: object
    """
    with open(file + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

