#! /usr/bin/env python

import tensorflow as tf
import data_helpers
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
checkpoint_file = os.path.join("dataset1_L_update3")
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# load the data from the test data
# x_test : array that [data size, sequence length]
# y: labels of the correct class
# W: vector
seq_size = 350
classes = 871
x_test,y,W = data_helpers.create_test_data("dataset1_test_recent_data",seq_size)
batch_size = 128


print("\nEvaluating...\n")

# Evaluation
# ==================================================


graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        embedding_placeholder = graph.get_operation_by_name("embedding_placeholder").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Generate batches for one epoch
        # batches = data_helpers.batch_iter(list(zip(x_test,y)), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here

        """
         total = np.empty((0, classes), int)
        total_batch = x_test.shape[0]/batch_size
        print "there is {:d} iteration to run".format(total_batch)
        for batch in range(total_batch):
            print "the {:d} time iteration begin..".format(batch)
            if batch == total_batch -1:
                x_batch, y_batch = x_test[batch * batch_size:x_test.shape[0]], y[batch * batch_size:x_test.shape[0]]
            else:
                x_batch, y_batch = x_test[batch * batch_size:(batch + 1) * batch_size], y[batch * batch_size:(batch+ 1) * batch_size]
            batch_predictions = sess.run(scores, {input_x: x_batch, dropout_keep_prob: 1.0, embedding_placeholder: W})
            # output the rank of correct prediction
            total = np.vstack((total, np.array(batch_predictions)))
        ranks = (-np.array(total)).argsort(axis=1).argsort(axis=1)
        # result = [np.where(ranks[index] == y[index])[0][0] for index in range(len(y))]
        print ranks


        with open('cnn_top50.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            total_batch = x_test.shape[0]/batch_size
            print "there is {:d} iteration to run".format(total_batch)
            for batch in range(total_batch):
                print "the {:d} time iteration begin..".format(batch)
                if batch == total_batch -1:
                    x_batch, y_batch = x_test[batch * batch_size:x_test.shape[0]], y[batch * batch_size:x_test.shape[0]]
                else:
                    x_batch, y_batch = x_test[batch * batch_size:(batch + 1) * batch_size], y[batch * batch_size:(batch+ 1) * batch_size]
                batch_predictions = sess.run(scores, {input_x: x_batch, dropout_keep_prob: 1.0, embedding_placeholder: W})
                ranks = (-np.array(batch_predictions)).argsort(axis=1)
                for index, rank in enumerate(ranks):
                    nums = np.where(ranks[index] == y_batch[index])[0].tolist()
                    print nums
                    nums.extend(rank[:50])
                # output the rank of correct prediction
                    spamwriter.writerow(nums)

        """

        total = []
        total_batch = x_test.shape[0]/batch_size
        print "there is {:d} iteration to run".format(total_batch)
        for batch in range(total_batch):
            print "the {:d} time iteration begin..".format(batch)
            if batch == total_batch -1:
                x_batch, y_batch = x_test[batch * batch_size:x_test.shape[0]], y[batch * batch_size:x_test.shape[0]]
            else:
                x_batch, y_batch = x_test[batch * batch_size:(batch + 1) * batch_size], y[batch * batch_size:(batch+ 1) * batch_size]
            batch_predictions = sess.run(scores, {input_x: x_batch, dropout_keep_prob: 1.0, embedding_placeholder: W})
            ranks = (-np.array(batch_predictions)).argsort(axis=1)
            # output the rank of correct prediction
            total.extend([np.where(ranks[index] == y_batch[index])[0][0] for index in range(len(y_batch))])


        print total
        h = sorted(total)  # sorted
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


