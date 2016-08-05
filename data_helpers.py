import numpy as np
import re
import random
import pickle

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    print "before the filter, the data size is :{:d}".format(len(string))
    string = ''.join([i if ord(i)>=32 and ord(i) < 127 else ' ' for i in string])
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"([\.=#]){2,}"," ",string)
    string = re.sub(r"(([\.=#])\s?){2,}\s"," ",string)
    # remove all the words that more than 20 characters (like website, some numbers combination, etc)
    string = re.sub(r"(OBJ_NUM[ ]+){2,}","OBJ_NUM ",string)
    # only keep one obj_num for continuously words
    string = re.sub(r"\s(\d){5,}\s", " ",string)
    # remove all the numbers that more than 4 digits
    string = re.sub(r"\s((\d)+([^\s\w])+(\d)+)+\s"," ",string)
    # remove all the combination of numbers and special character
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s[\w]{18,}\s"," ",string)
    # remove all the space
    # remove special characters except: ():_@/\.\'
    print "after the filter, the data size is :{:d}".format(len(string))
    return string

def separate_data():
    """
    separate the data according to the userID
    :return: None
    """
    with open("training_data_engineer_note_old.txt","r") as infile,open("train1.txt","w") as file1, open("train2.txt","w") as file2:
        for line in infile:
            if int(line.split(" ")[0]) < 430:
                file1.write(line)
            else:
                file2.write(line)


def create_sample_data():
    """
    create some fake examples that will work
    :return: None
    """
    choices = ["Company","EducationalInstitution","Artist","Athlete","OfficeHolder",\
               "MeanOfTransportation","Building","NaturalPlace", "Village",\
                "Animal", "Plant","Album","Film","WrittenWork"]
    with open("data.txt", "r") as infile, open("Output.txt", "w") as outfile:
            for line in infile:
                outfile.write(random.choice(choices) + " " + clean_str(line) + "\n")

def create_from_csv():
    """
    create examples from wikipedia data
    :return:  None
    """
    with open("train.csv", "r") as infile,open("output2.txt", "w") as outfile:
        for line in infile:
            choice = line.split(",")[0]
            text = "".join(line.split(",")[2:])
            outfile.write(choice + " " + clean_str(text) + "\n")


def load_engineer_name(total):
    """
    load the engineer name into the dictionary
    :param fileName: file that contains the engineer name in each line
    :return: engineer name and index pair
    """
    """
     with open(engFile, "r") as f:
        nameList = f.read().splitlines()
    total = len(nameList)
    dictionary = {i+1: [0]*i+[1]+[0]*(total-i-1) for i, x in enumerate(nameList)}
    """
    dictionary = {i: [0] * i + [1] + [0] * (total - i - 1) for i in range(total)}
    return dictionary


def load_data_from_glove(gloveFile):
    """
    load the data from files, split the data into vectors and labels
    :return: list, first part is the word, the second part is the vector
    """
    with open(gloveFile,"r") as f:
        word_dic = {line.split()[0]: np.asarray(line.split()[1:], dtype='float') for line in f}
    return word_dic


def load_data_and_labels(examples,engineer,seq_size):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    input : txt file of pos and neg examples
    output: list, first part is the vectors, the second part is the labels
    """
    # Load data from files
    examples = list(open(examples, "r").readlines())
    examples = [s.strip() for s in examples]
    # calculate how many classes it has, now cuz some problem, one person has two line.
    total = sum(1 for _ in open(engineer)) -2
    print "there are " + str(total) + " classes"
    engineer_dictionary = load_engineer_name(total)
    save_obj(engineer_dictionary, "engineer")
    labels = [engineer_dictionary[int(case.split()[0])] for case in examples]
    x_text = [s.split(" ")[1:seq_size+1] for s in examples]
    return [x_text, labels]


def pad_sentences(sentences, padding_word="."):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    print "for whole document, the maximum length is: " + str(sequence_length)
    print "the document length is: " + str(len(sentences))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(gloveFile):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    vocab = []
    vocab_index = {}
    with open(gloveFile, "r") as f:
        for index, line in enumerate(f):
            word = line.split()[0]
            vector = np.asarray(line.split()[1:], dtype='float')
            vocab.append(vector)
            vocab_index[word] = index
    return [vocab_index, vocab]


def build_input_data(sentences, labels, vocabulary_index):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary_index.get("unknown") if vocabulary_index.get(word) is None \
                   else vocabulary_index.get(word) for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def create_test_data(file,seq_size):
    # Load data from files
    examples = list(open(file, "r").readlines())
    examples = [s.strip() for s in examples]
    # engineer_dictionary = load_obj("engineer")
    labels = [int(case.split()[0]) for case in examples]
    x_text = [s.split(" ")[1:seq_size+1] for s in examples]
    x_text = pad_sentences(x_text)
    vocabulary_index = load_obj("index")
    vocabulary = load_obj("vocabulary")
    x,y = build_input_data(x_text,labels,vocabulary_index)
    return [x,y,vocabulary]


def create_retrain_data(file):
    # Load data from files
    examples = list(open(file, "r").readlines())
    examples = [s.strip() for s in examples]
    engineer_dictionary = load_obj("engineer")
    labels = [engineer_dictionary[int(case.split()[0])] for case in examples]
    x_text = [s.split(" ")[1:] for s in examples]
    x_text = pad_sentences(x_text)
    vocabulary_index = load_obj("index")
    vocabulary = load_obj("vocabulary")
    x,y = build_input_data(x_text,labels,vocabulary_index)
    return [x,y,vocabulary]


def load_data(train,output,vector,seq_size):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and pre-process data
    sentences, labels = load_data_and_labels(train, output, seq_size)
    sentences_padded = pad_sentences(sentences)
    vocabulary_index, vocabulary = build_vocab(vector)
    save_obj(vocabulary_index,"index")
    save_obj(vocabulary,"vocabulary")
    x, y = build_input_data(sentences_padded, labels, vocabulary_index)
    return [x, y, vocabulary, vocabulary_index]

def load_more_data(train,output,seq_size, vocabulary,vocabulary_index):
    sentences, labels = load_data_and_labels(train,output,seq_size)
    sentences_padded = pad_sentences(sentences)
    x, y = build_input_data(sentences_padded, labels, vocabulary_index)
    return [x, y]


def save_obj(obj, name):
    """
    save all the model in the pickle
    :param obj: Object
    :param name: file Name
    :return: None
    """
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    load the model from the pickle
    :param name: file Name
    :return: object
    """
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    total iteration: num_epochs * (len(data)/batch_size +1)
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = min(batch_num * batch_size, data_size-1)
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

