import tensorflow as tf
import datetime
import numpy as np
from CharCNN import CharCNN
from data_tool import data_tool

# basic configuration
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


# Training procedures
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