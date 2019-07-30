import tensorflow as tf
import os


def lab_04_1():
    x1_data = [73., 93., 89., 96., 73.]
    x2_data = [80., 88., 91., 98., 66.]
    x3_data = [75., 93., 90., 100., 70.]

    y_data = [152., 185., 180., 196., 142.]

    x1 = tf.compat.v1.placeholder(tf.float32)
    x2 = tf.compat.v1.placeholder(tf.float32)
    x3 = tf.compat.v1.placeholder(tf.float32)

    Y = tf.compat.v1.placeholder(tf.float32)

    w1 = tf.Variable(tf.random.normal([1]), name='weight1')
    w2 = tf.Variable(tf.random.normal([1]), name='weight2')
    w3 = tf.Variable(tf.random.normal([1]), name='weight3')
    b = tf.Variable(tf.random.normal([1]), name='bias')

    hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(10001):
        w1_val, w2_val, w3_val, b_val, cost_val, hy_val, _ = sess.run([w1, w2, w3, b, cost, hypothesis, train],
                                       feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})

        if step % 10 == 0:
            print("w1: ", w1_val, "\nw2: ", w2_val, "\nw3: ", w3_val, "\nb: ", b_val)
            print(step, "Cost: ", cost_val, "\nhypothesis:\n", hy_val)
            print()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sess = tf.compat.v1.Session()

    lab_04_1()
