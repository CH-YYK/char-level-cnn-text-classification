import tensorflow as tf
import numpy as np
import pandas as pd
import re, os
import datetime

#train_path = "data/train.tsv"
#test_path = "data/test.tsv"


train_path = "data/train.csv"
test_path = "data/test.csv"
truncated_length = 1014
conv_config = [[7, 256, 3],
               [7, 256, 3],
               [3, 256, None],
               [3, 256, None],
               [3, 256, None],
               [3, 256, 3]]

fc_config = [1024, 1024]


class data_tool(object):

    def __init__(self, train_path, test_path, truncated_length):
        self.train_path = train_path
        self.test_path = test_path

        print("load data...")
        # training set
        # self.train = pd.read_table(train_path, sep="\t")
        # self.train_x = [i[:truncated_length] for i in self.train['Phrase']]
        self.train = pd.read_csv(train_path, names=['1', '2', '3'])
        self.train_x = [' '.join(self.train.iloc[i, [1, 2]])[:truncated_length] for i in range(self.train.shape[0])]

        # self.train_y = [[0] * 4 for i in range(self.train.shape[0])]
        # _ = [self.train_y[i].insert(j, int(j)) for i, j in enumerate(self.train['Sentiment'])]
        self.train_y = [[0] * 3 for i in self.train.iloc[:, 0]]
        _ = [self.train_y[i].insert(j-1, 1) for i, j in enumerate(self.train.iloc[:, 0])]


        # test set
        #self.test = pd.read_table(test_path, sep='\t')
        #self.test_x = [i[:truncated_length] for i in self.test['Phrase']]

        self.test = pd.read_csv(test_path, names=['1', '2', '3'])
        self.test_x = [' '.join(self.test.iloc[i, [1, 2]])[:truncated_length] for i in range(self.test.shape[0])]
        print("data loaded!")

        self.test_y = [[0] * 3 for i in self.test.iloc[:, 0]]
        _ = [self.test_y[i].insert(j - 1, 1) for i, j in enumerate(self.test.iloc[:, 0])]
        self.test_y = np.array(self.test_y)

        # character_dict
        self.char_dict = self.character_corpus()

        # word_vector
        self.one_hot_word_vector = self.to_one_hot(self.char_dict, truncated_length)

        # form data
        self.train_x = np.array([self.text2index(j, self.char_dict, truncated_length) for j in self.train_x])
        self.test_x = np.array([self.text2index(j, self.char_dict, truncated_length) for j in self.test_x])

        self.train_y = np.array(self.train_y)
        self.test_y = np.array(self.test_y)

    def clean(self, string):
        return re.sub("[a-zA-Z]\\+[a-zA-Z]", '\n', string)

    def character_corpus(self):
        char_dict = {char: index + 1 for index, char
                     in enumerate("abcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'\"/\\|_@#$%^&*~`+-=<>()[]{}")}
        char_dict['\n'] = 69
        return char_dict

    def to_one_hot(self, char_dict, truncated_length):
        tmp = np.zeros([truncated_length, char_dict.keys().__len__()])
        for i, j in enumerate(char_dict.values()):
            tmp[i, j-1] = 1
        return np.concatenate([np.zeros([truncated_length, 1]), tmp], axis=1)

    def text2index(self, text, vocab_dict, truncated_length):
        """
        tokenization
        """
        text = self.clean(text)
        tmp = [0] * truncated_length
        for i in range(1, len(text)+1):
            tmp[i-1] = vocab_dict.get(text[-i].lower(), 0)
        return tmp

    # generate batches of data to train
    def generate_batches(self, data_x, data_y, epoch_size, batch_size, shuffle=False):
        data_size = len(data_x)
        num_batches = data_size // batch_size + 1

        for i in range(epoch_size):
            if shuffle:
                np.random.seed(1000)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_data_x, shuffle_data_y = data_x[shuffle_indices], data_y[shuffle_indices]
            else:
                shuffle_data_x, shuffle_data_y = data_x, data_y

            for j in range(num_batches):
                start_index = j * batch_size
                end_index = min((j+1) * batch_size, data_size)
                batch_x = shuffle_data_x[start_index: end_index]
                batch_y = shuffle_data_y[start_index: end_index]
                yield batch_x, batch_y

    def save_data(self, result):
        test_data = pd.read_csv(test_path, names=['1', '2', '3'])
        test_data['4'] = [i+1 for i in result.reshape(-1).tolist()]
        test_data = test_data.loc[:, ['1', '4']]
        print((test_data['1'] == test_data['4']).mean())
        test_data.to_csv("data/sample_submission.csv", index=False)


class CharCNN(object):

    def __init__(self, sequence_length, conv_config, fc_config, char_vector, num_classes=5):
        l2_loss = 0
        # define Placeholders
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_x')
        self.label_y = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name='label')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        #
        with tf.name_scope("character_embedding"):
            W = tf.Variable(dtype=tf.float32, initial_value=char_vector,trainable=False,
                            name="character_embedding")
            self.input_ = tf.nn.embedding_lookup(W, self.input_x, name='character_embedding')
        #
        with tf.name_scope("cnn_pooling_stacks"):
            self.input_ = tf.cast(tf.expand_dims(self.input_, axis=-1), "float")
            for index, config in enumerate(conv_config):
                cnn_config = [self.input_] + config + [index]
                self.input_ = self.cnn_maxpool(*cnn_config)


        # flatten the result
        total_neurons = self.input_.get_shape()[1].value * self.input_.get_shape()[2].value
        self.input_ = tf.reshape(self.input_, shape=[-1, total_neurons])

        with tf.name_scope("fully_connected"):
            for index, num_out in enumerate(fc_config):
                self.input_ = self.fc_layers(self.input_, num_out, index)

        with tf.name_scope("scores_and_output"):
            width = self.input_.get_shape()[1].value
            stddev = 1/(width**1/2)
            w_out = tf.Variable(tf.random_uniform([width, num_classes], minval=-stddev, maxval=stddev),
                                name="output_weight")
            b_out = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-stddev, maxval=stddev),
                                name="b")

            l2_loss += tf.nn.l2_loss(w_out)
            l2_loss += tf.nn.l2_loss(b_out)

            self.scores = tf.nn.xw_plus_b(self.input_, w_out, b_out, name='output_bias')
            self.softmax_scores = tf.nn.softmax(self.scores)
            self.output = tf.argmax(self.scores, axis=1)

        with tf.name_scope("loss_and_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label_y)
            self.loss = tf.reduce_mean(losses)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.label_y, axis=1)), "float"))



    def cnn_maxpool(self, input_, kernel_size, num_filters, pooling_size, index):
        with tf.name_scope("cnn_maxpool_%s" % index):
            embedding_length = input_.get_shape()[2].value

            filter_size = [kernel_size, embedding_length, 1, num_filters]

            # W = tf.get_variable(name="cnn_weights_%s" % index,
            #                     initializer=tf.random_normal(shape=filter_size, mean=0, stddev=0.05))
            # b = tf.get_variable(name='cnn_bias_%s' % index,shape=[num_filters],
            #                    initializer=tf.random_normal_initializer(mean=0, stddev=0.05))


            # an alternative initializer
            stdv = 1 / ((kernel_size * embedding_length) ** 1/2)
            W = tf.Variable(tf.random_uniform(filter_size, minval=-stdv, maxval=stdv), dtype='float32',
                            name="cnn_weight_%s" % index)  # The kernel of the conv layer is a trainable vraiable
            b = tf.Variable(tf.random_uniform([num_filters], minval=-stdv, maxval=stdv),
                            name='cnn_b_%s' % index)

            # convolution
            conv = tf.nn.conv2d(input_, W, strides=[1, 1, 1, 1], padding='VALID', name='conv_%s' % index)


            # apply non-linear
            h = tf.nn.bias_add(conv, b)
            if pooling_size:
                pool = tf.nn.max_pool(h, ksize=[1, pooling_size, 1, 1], strides=[1, pooling_size, 1, 1], padding='VALID')
                return tf.transpose(pool, perm=[0, 1, 3, 2])
            else:
                return tf.transpose(h, perm=[0, 1, 3, 2])

    def fc_layers(self, input_, num_outputs, index):
        with tf.name_scope("fully_connected_layer_%s" % index):
            width = input_.get_shape()[1].value
            stddev = 1/(width**1/2)
            W = tf.get_variable(name="fully_connected_weight_1_%s" % index,
                                initializer=tf.random_uniform([width, num_outputs], minval=-stddev, maxval=stddev))
            b = tf.get_variable(name="fully_connected_bias_2_%s" % index,
                                initializer=tf.random_uniform([num_outputs], minval=-stddev, maxval=stddev))

            h = tf.nn.xw_plus_b(input_, W, b)

            return tf.nn.dropout(h, keep_prob=self.keep_prob)




class Training(data_tool, CharCNN):

    def __init__(self):
        self.batch_size = 128
        self.epoch_size = 15
        data_tool.__init__(self, train_path, test_path, truncated_length=truncated_length)
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                CharCNN.__init__(self, sequence_length=truncated_length, conv_config=conv_config, fc_config=fc_config,
                                 char_vector=self.one_hot_word_vector, num_classes=4)

                global_step = tf.Variable(0, name='global_step', trainable=False)

                self.saver = tf.train.Saver()

                optimizer = tf.train.AdamOptimizer(0.001)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step)
                # train_op = optimizer.minimize(self.loss)

                # Keep track of gradient values and sparsity

                # Summary for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join("runs", "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Test Summaries
                test_summary_op = tf.summary.merge([loss_summary, acc_summary])
                test_summary_dir = os.path.join('runs', 'summaries', 'test')
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

                # define operations
                def train_(batch_x, batch_y):
                    feed_dict = {self.input_x: batch_x,
                                 self.label_y: batch_y,
                                 self.keep_prob: 0.5}

                    loss, _, accuracy, step, summaries = sess.run(
                        [self.loss, train_op, self.accuracy, global_step, train_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)


                def test_(indices):
                    feed_dict = {self.input_x: self.test_x[indices[:1000]],
                                 self.label_y: self.test_y[indices[:1000]],
                                 self.keep_prob: 1.0}

                    loss, accuracy, step, summaries = sess.run(
                        [self.loss, self.accuracy, global_step, test_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    test_summary_writer.add_summary(summaries, step)

                # initialize variable
                sess.run(tf.global_variables_initializer())

                # generate batches
                batches_all = self.generate_batches(data_x=self.train_x, data_y=self.train_y, epoch_size=self.epoch_size,
                                                    batch_size=self.batch_size, shuffle=True)
                total_amount = (len(self.train_x) // self.batch_size + 1) * self.epoch_size

                # generate test indices
                shuffle_indices = np.random.permutation(np.arange(len(self.test_x)))

                # training on batches
                print("Total step:", total_amount)
                for i, batch in enumerate(batches_all):
                    batch_x, batch_y = batch
                    train_(batch_x, batch_y)

                    if i % 100 == 0:
                        print('\nEvaluation:\n')
                        test_(shuffle_indices)

                        print("Writing model...\n")
                        self.saver.save(sess, "tmp/model.ckpt", global_step=1)

                # start testing and saving data
                data_size = len(self.test_x)
                result = []
                for i in range(data_size // 500):
                    result.append(sess.run(self.output,
                                           feed_dict={self.input_x: self.test_x[i * 500:(i + 1) * 500],
                                                      self.keep_prob: 1.0}))
                result.append(sess.run(self.output,
                                       feed_dict={self.input_x: self.test_x[(i + 1) * 500:],
                                                  self.keep_prob: 1.0}))
                self.result = np.concatenate(result, axis=0)

                # save result to file
                self.save_data(self.result)

if __name__ == '__main__':
    Training()