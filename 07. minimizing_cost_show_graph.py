
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def lab_03_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    W = tf.compat.v1.placeholder(tf.float32)

    hypothesis = x * W

    cost = tf.reduce_mean(tf.square(hypothesis - y))
    sess.run(tf.compat.v1.global_variables_initializer())

    W_val = []
    cost_val = []

    for i in range(-30, 50):
        feed_W = i * 0.1
        curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
        print("Cost: ", curr_cost, "\nW: ", curr_W, "\n")
        W_val.append(curr_W)
        cost_val.append(curr_cost)

    plt.plot(W_val, cost_val)
    plt.show()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sess = tf.compat.v1.Session()

    lab_03_1()
