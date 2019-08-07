
import tensorflow as tf
import os


def softmax_classifier():
    x_data = [[1, 2, 1, 1],
              [2, 1, 3, 2],
              [3, 1, 3, 4],
              [4, 1, 5, 5],
              [1, 7, 5, 5],
              [1, 2, 5, 6],
              [1, 6, 6, 6],
              [1, 7, 7, 7]]
    y_data = [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 0]]

    X = tf.compat.v1.placeholder('float', [None, 4])
    Y = tf.compat.v1.placeholder('float', [None, 3])

    # 분류할 클래스 수
    nb_classes = 3

    W = tf.compat.v1.Variable(tf.random.normal([4, nb_classes]), name='weight')
    b = tf.compat.v1.Variable(tf.random.normal([nb_classes], name='bias'))

    hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(X, W) + b)

    cost = tf.compat.v1.reduce_mean(-tf.compat.v1.reduce_sum(Y * tf.compat.v1.log(hypothesis), axis=1))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(10001):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

        hy_val, cost_val = sess.run([hypothesis, cost], feed_dict={X: x_data, Y: y_data})

        # 예측 결과
        print(hy_val)

        # one-hot encoding
        print(sess.run(tf.math.argmax(hy_val, 1)))

        # 최종적으로 도달한 코스트값
        print(cost_val)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    softmax_classifier()
