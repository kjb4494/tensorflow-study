import tensorflow as tf
import os


# tensorflow reader로 파일을 읽어서 메모리 낭비를 줄임

def lab_04_4():
    filename_queue = tf.train.string_input_producer(
        ['data-01-test-score.csv'], shuffle=False, name='filename_queue'
    )
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    record_defaults = [[0.], [0.], [0.], [0.]]
    xy = tf.io.decode_csv(value, record_defaults=record_defaults)

    train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    W = tf.Variable(tf.random.normal([3, 1]), name='weight')
    b = tf.Variable(tf.random.normal([1]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess.run(tf.compat.v1.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch}
        )
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sess = tf.compat.v1.Session()

    lab_04_4()
