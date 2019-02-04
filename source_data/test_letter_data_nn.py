import tensorflow as tf
import numpy as np
from copy import copy
from random import shuffle


def load_letter_data():
    letter_data = []
    label_data = []

    labels = []
    features_max = []

    print('+ Load data ...')
    with open('data/letter-recognition.data', 'r') as iris_file:
        for line in iris_file.readlines():
            cur_line = [elm.strip() for elm in line.split(',')]

            if len(cur_line) == 17:
                cur_label = cur_line[0]
                if cur_label not in labels:
                    labels.append(cur_label)

                label_data.append(labels.index(cur_label))

                features = [float(elm) for elm in cur_line[1:]]
                if len(features_max) == 0:
                    features_max = [elm for elm in features]
                else:
                    for idx, feature in enumerate(features):
                        if features_max[idx] < feature:
                            features_max[idx] = feature

                letter_data.append(features)
    
    features_max = np.array(features_max, np.float64)
    letter_data = np.divide(np.array(letter_data, np.float64), features_max)
    ##
    # expand labels (one hot vector)
    tmp = np.zeros((len(label_data), len(labels)))
    tmp[np.arange(len(label_data)), label_data] = 1
    label_data = tmp

    print('+ letters: \n', letter_data)
    print('+ labels: \n', label_data)

    print('+ loading done!')
    return letter_data, label_data


def batch(data, label, size):
    out_data = []
    out_label = []
    for index, elm in enumerate(data):
        if len(out_data) < size:
            out_data.append(elm)
            out_label.append(label[index])
        else:
            yield out_data, out_label
            out_data = []
            out_label = []


def main():
    images_data, label_data = load_letter_data()

    train_percentage = 0.8

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    train_size = int(len(images_data) * train_percentage)
    train_count = 0
    num_round = 1

    indexes = [_ for _ in range(len(images_data))]

    for round_ in range(num_round):
        shuffle(indexes)
        #print("+ indexes", indexes)
        for index in indexes:
            if train_count < train_size:
                train_data.append(copy(images_data[index]))
                train_labels.append(copy(label_data[index]))
                train_count += 1
            else:
                test_data.append(copy(images_data[index]))
                test_labels.append(copy(label_data[index]))
        train_count = 0

    print("+ train size:", len(train_data))
    print("+ test size:", len(test_data))

    LABEL_SIZE = len(train_labels[0])
    FEATURE_SIZE = len(train_data[0])

    # Create the model
    x = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
    W = tf.Variable(tf.zeros([FEATURE_SIZE, LABEL_SIZE]))
    b = tf.Variable(tf.zeros([LABEL_SIZE]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()

    step_size = 4
    for batch_xs, batch_ys in batch(train_data, train_labels, step_size):
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

     # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("+ Accuracy: ", sess.run(accuracy, feed_dict={x: test_data,
                                                        y_: test_labels}))

    print("+ W:\n{}".format(sess.run(W)))
    print("+ b:\n{}".format(sess.run(b)))

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
