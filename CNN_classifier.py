# Yao Zhou
# July 20, 2016

import tensorflow as tf
import data_helpers_cnn
import numpy as np
import matplotlib.pyplot as plt
import time, os

class CNNClassifier(object):
    """
    CNN classifier for create CNN model instance
    """
    datahelpers = data_helpers_cnn  # Instance of ./DataHelpers
    batch_size = 128
    allow_soft_placement = True
    log_device_placement = False
    checkpoint_file = None
    seq_size = 0
    total_class = 0
    pickle_file = None
    graph = None
    session = None
    input_x = None
    dropout_keep_prob = 0
    embedding_placeholder = None
    scores = None
    vocabulary_index = None
    vocabulary = None
    count = 0
    accuracy = 0
    note_accuracy = 0
    note_count = 0


    def __init__(self,pickle_file, model, seq_size= 350, total_class=871, batch_size = 1, allow_soft_placement = True, log_device_placement = False):
        """
        :param
        pickle_file: obj file position
        model: cnn result
        seq_size: how long the test file need to be truncated
        total_class: y length
        batch_size: process how many data at one time
        allow_soft_placement: whether can change between GPU & CPU machine
        """
        self.batch_size = batch_size
        self.pickle_file = pickle_file
        self.checkpoint_file = os.path.join(model)
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.seq_size = seq_size
        self.total_class = total_class
        self.vocabulary_index = self.datahelpers.load_obj(self.pickle_file,"index")
        self.vocabulary = self.datahelpers.load_obj(self.pickle_file,"vocabulary")



    def load_model(self):
        """
        create graph and session for test data
        """
        self.graph = tf.Graph()
        # NUM_THREADS = 21
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement)
            self.session = tf.Session(config=session_conf)

            with self.session.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.session, self.checkpoint_file)

                # print_tensors_in_checkpoint_file(checkpoint_file, "embedding")
                # print_function(checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
                self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                self.embedding_placeholder = self.graph.get_operation_by_name("embedding_placeholder").outputs[0]
                # Tensors we want to evaluate
                self.scores = self.graph.get_operation_by_name("output/scores").outputs[0]


    def load_result(self,cases,label):
        """
        :param notes: list of strings
        :return: len(notes) x n(classes) Numpy array of float probabilities corresponding to classes
                 Probabilities vary from 0 to 1, should sum to 1 for each note
        """
        self.count += 1
        print "{:d}, total {:d} cases".format(self.count,len(cases))
        x_test = self.datahelpers.create_case_data(cases, self.seq_size, self.vocabulary_index)

        def softmax(array):
            array = np.exp(array)
            array /= np.sum(array)
            return array

        with self.graph.as_default():
            with self.session.as_default():
                predictions = self.session.run(self.scores,
                                             {self.input_x: x_test, self.dropout_keep_prob: 1.0, self.embedding_placeholder: self.vocabulary})
                predictions = np.array([softmax(prediction) for prediction in predictions])
                for prediction in predictions:
                    self.note_count += 1
                    if int(label) == np.argmax(prediction):
                        self.note_accuracy += 1
                        print "correct for {:d} case, note accuracy is {:f}".format(self.note_count,float(self.note_accuracy)/self.note_count)
                return predictions

    def predict_proba_case(self, case_notes, label):
        """
        :param case_notes: List of tuples (int(note_type), str(note_text)) for each note in the case
        :return: 1D n_class long np array of class probabilities
        """
        # Convert case_notes into X_test (based on notes) and weights (based on note types)
        weights = np.ones(len(case_notes), dtype=np.float_)
        notes = []
        for i, tup in enumerate(case_notes):
            note_type, feature, note = tup
            weights[i] = note_type
            notes.append(note)
        weights = weights / np.sum(weights)
        prob_mat = self.load_result(notes,label)
        if prob_mat.shape[0] == 1:
            probs = prob_mat[0]
        else:
            probs = np.average(prob_mat, axis=0, weights=weights)
        if int(label) == np.argmax(probs):
            self.accuracy += 1
            print "predict correctly for {:d} case, accuracy: {:2f}".format(self.count, float(self.accuracy)/self.count)
        return probs


    def predict_correct_rank(self,total,y):
        """
        :param result:
        :return: the rank for the real engineer who solves the case
        """
        ranks = (-np.array(total)).argsort(axis=1)
        result = [np.where(ranks[index] == y[index])[0][0] for index in range(len(y))]
        return result


    def predict_ranks(self, total):
        """
        :param notes:
        :return: len(notes) x n(classes) Numpy array of integer ranks corresponding to classes.
                 Ranks vary from 0 to 870
        """
        return (total).argsort(axis=1).argsort(axis=1)


    def predict_single_rank(self,total):
        """
        :param total: a single note probability
        :return: one line Numpy array of integer ranks corresponding to classes.
                 Ranks vary from 0 to 870
        """
        return (total).argsort().argsort()


    def create_accuracy_plot(self,result):
        """
        :param result: the accuracy list
        :return: None
        """
        h = sorted(result)  # sorted
        # fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
        # plt.plot(h, fit, '-.')
        # plt.hist(h, bins = 40,normed=True)  # use this to draw histogram of your data
        # plt.show()

        n_bins = 850
        n, bins, patches = plt.hist(h, n_bins, normed=1,
                                    histtype='step', cumulative=True)
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.xlim(0, 100)
        plt.title('cumulative step')
        plt.show()
