import tensorflow as tf
import os
import numpy as np


def lab_05_1():
    xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    W = tf.compat.v1.Variable(tf.random.normal([8, 1]), name='weight')
    b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')

    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    cost = -tf.reduce_mean(Y * tf.math.log(hypothesis) + (1 - Y) * tf.math.log(1 - hypothesis))

    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

    predicted = tf.compat.v1.cast(hypothesis > 0.5, dtype=tf.float32)

    # 예측값과 실제 값이 맞았는지 리포팅
    accuracy = tf.compat.v1.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.compat.v1.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(10001):
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
            if step % 200 == 0:
                print(step, cost_val)

        # Accuracy report
        h, c, a = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: x_data, Y: y_data})
        print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    lab_05_1()
