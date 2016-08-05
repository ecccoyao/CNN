#! /usr/bin/env python

import tensorflow as tf
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import numpy as np


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 871, "number of classes in the data")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# train data: the note, first word is the class, the others are note
# output data: the engineer name list
# train again with other data
output = "support_engineers.csv"
# vector: glove & word2vec
train = "dataset1_clean_1"
vector = "w2v.txt"
# seq_size: how many data should people keep
seq_size = 350
# number of iteration through the data
num_of_round = 110

x, y, W, vocabulary = data_helpers.load_data(train,output,vector,seq_size)

# Training
# ==================================================

with tf.Graph().as_default():
    # A Graph contains operations and tensors.
    # Session is the environment you are executing graph operations in, and it contains state about Variables and queues.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x.shape[1],
            # the length of the sentence
            num_classes=FLAGS.num_classes,
            # how many classes as output
            vocab_size=len(vocabulary),
            # total vocabulary
            embedding_size=FLAGS.embedding_dim,
            # vector length
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            # map function: Apply function to every item of iterable and return a list of the results
            # k filters (or kernels) of size n x n x q
            # where nn is smaller than the dimension of the image and qq can either be the same as the number of channels rr or smaller and may vary for each kernel.
            # each map is then subsampled typically with mean or max pooling over p x pp x p contiguous regions
            # where p ranges between 2 for small images (e.g. MNIST) and is usually not more than 5 for larger inputs.
            num_filters=FLAGS.num_filters,
            # how many filters in one layer
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # allow TensorFlow handle the counting of training steps for us
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # train_op here is a newly created operation that we can run to perform a gradient update on our parameters
        # TensorFlow automatically figures out which variables are trainable and calculates their gradients
        # global step will be automatically incremented by one every time you execute train_op

        # Keep track of gradient values and sparsity (optional)
        # keep track of how your loss and accuracy evolve over time.
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.embedding_placeholder: W,
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.embedding_placeholder: W,
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        """
        # Generate cross validation data and batches
        cv = cross_validation.KFold(len(x), n_folds=len(x)/500, indices=False)
        print len(x)/10
        print "start cross validation"
        times = 0
        for traincv, testcv in cv:
            if times < 10:
                times += 1
                print "start the " + str(times) + " cross validation cycle..."
                x_train, x_dev = x[traincv], x[testcv]
                y_train, y_dev = y[traincv], y[testcv]
                batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                for batch in batches:
                    try:
                        x_batch, y_batch = zip(*batch)
                        train_step(x_batch, y_batch)
                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % FLAGS.evaluate_every == 0:
                            print("\nEvaluation:")
                            dev_step(x_dev, y_dev, writer=dev_summary_writer)
                            print("")
                        if current_step % FLAGS.checkpoint_every == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                    except:
                        print "error"
            else:
                print "finish the training cycle. Wait for evaluation"
                break
        """

        # start training
        for i in range(num_of_round):
            seq = str(i%7) + 1
            print "start to train with the " +  str(i+1) + " round of the data"
            x, y = data_helpers.load_more_data("dataset1_clean_"+seq,output, seq_size, W, vocabulary)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            # Split train/test set
            # TODO: This is very crude, should use cross-validation
            x_train, x_dev = x_shuffled[:-100], x_shuffled[-100:]
            y_train, y_dev = y_shuffled[:-100], y_shuffled[-100:]
            print("Vocabulary Size: {:d}".format(len(vocabulary)))
            print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                except:
                    print "error"

