import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self,sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        with tf.device('/gpu:1'):
            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                            trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size],name="embedding_placeholder")
            embedding_init = W.assign(self.embedding_placeholder)


            # print embedding_init.eval()[0]
            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Embedding layer
            with tf.name_scope("embedding"):
                self.embedded_chars = tf.nn.embedding_lookup(embedding_init, self.input_x)
                #The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size].
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
                # training data shape: 1 for the channel [None, sequence_length, embedding_size, 1]

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # create a layer for each of them, and then merge the results into one big feature vector
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    # each convolution produces tensors of different shapes we need to iterate through them, create a layer for each of them, and then merge the results into one big feature vector.
                    # convolutional conv2d operation expects a 4-dimensional tensor with dimensions corresponding to batch, width, height and channel
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    # Each filter slides over the whole embedding, but varies in how many words it covers
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        # slide the filter over our sentence without padding the edges, performing a narrow convolution
                        name="conv")
                    # n_{out}=(n_{in} + 2*n_{padding} - n_{filter}) + 1 .
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    # W is our filter matrix and h is the result of applying the nonlinearity to the convolution output
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
                    # Performing max-pooling over the output of a specific filter size leaves us with a tensor of shape [batch_size, 1, 1, num_filters]
                    # the last dimension corresponds to our features

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(3, pooled_outputs)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            # Once we have all the pooled output tensors from each filter size
            # we combine them into one long feature vector of shape [batch_size, num_filters_total]
            # Using -1 in tf.reshape tells TensorFlow to flatten the dimension when possible

            # Add dropout
            # A dropout layer stochastically disables a fraction of its neurons.
            # This prevent neurons from co-adapting and forces them to learn individually useful features
            # 0.5 during training, and to 1 (disable dropout) during evaluation.
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (un-normalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                # we can generate predictions by doing a matrix multiplication and picking the class with the highest score.
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
                # can be updated by the softmax function to convert into 0-1 probability output

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
