
import tensorflow as tf
import os


def lab_04_2():
    x_data = [[73., 80., 75.],
              [93., 88., 93.],
              [89., 91., 90.],
              [96., 98., 100.],
              [73., 66., 70.]]
    y_data = [[152.],
              [185.],
              [180.],
              [196.],
              [142.]]

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    W = tf.compat.v1.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, W_val, b_val, _ = sess.run(
            [cost, hypothesis, W, b, train], feed_dict={X: x_data, Y: y_data}
        )
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nhypothesis:\n", hy_val, "\nW:\n", W_val, "\nb: ", b_val)
            print()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sess = tf.compat.v1.Session()

    lab_04_2()
