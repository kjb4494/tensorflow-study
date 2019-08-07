import tensorflow as tf
import os


def lab_02_1():
    x_train = [1, 2, 3]
    y_train = [3, 2, 1]

    W = tf.Variable(tf.random.normal([1]), name="weight")
    b = tf.Variable(tf.random.normal([1]), name="bias")

    hypothesis = x_train * W + b

    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(10001):
        sess.run(train)
        if step % 100 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))


# placeholder
def lab_02_2():
    a = tf.compat.v1.placeholder(tf.float32)
    b = tf.compat.v1.placeholder(tf.float32)
    adder_node = a + b
    print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
    print(sess.run(adder_node, feed_dict={a: [1, 2], b: [2, 4]}))


def lab_02_3():
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)

    x = tf.compat.v1.placeholder(tf.float32)
    y = tf.compat.v1.placeholder(tf.float32)

    hypothesis = x * W + b

    cost = tf.reduce_mean(tf.square(hypothesis - y))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(5000):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict={x: x_train, y: y_train}
        )
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nHypothesis:\n", hy_val)

    W_val, b_val, cost_val = sess.run([W, b, cost], feed_dict={x: x_train, y: y_train})
    print("W: ", W_val, "\nb: ", b_val, "\ncost: ", cost_val)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sess = tf.compat.v1.Session()

    # lab_02_1()
    # lab_02_2()
    lab_02_3()
