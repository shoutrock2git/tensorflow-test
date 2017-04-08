import numpy as np
import tensorflow as tf

NUM_DIGITS = 7

def binary_encode(i,num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def even_odd_encode(i):
    if i % 2 == 0:
        return np.array([1,0])
    else:
        return np.array([0,1])

trX = np.array([binary_encode(i,NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([even_odd_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

X = tf.placeholder(tf.float32,[None,NUM_DIGITS])
Y = tf.placeholder(tf.float32,[None,2])

NUM_HIDDEN = 5

w_h = init_weights([NUM_DIGITS,NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN,2])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=py_x))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1)

def even_odd(i, prediction):
    return ["{0:3d}" .format(i) + ":even", "{0:3d}" .format(i) + ":odd"][prediction]

BATCH_SIZE = 1

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(30):
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        if epoch % 10 == 0:
            print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    numbers = np.arange(1, 101)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(even_odd)(numbers, teY)

    print(output)

#
