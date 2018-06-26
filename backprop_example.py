import tensorflow as tf

# ---------------
# An implementation of the backpropagation example found at
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# ---------------

# network parameters
input_size = 2
hidden_size = 2
output_size = 2

# learning parameters
learning_rate = 0.5

# graph components
X = tf.placeholder(tf.float32, shape=[None, input_size], name='X')
Y = tf.placeholder(tf.float32, shape=[None, output_size], name='Y')
w1 = tf.Variable([[0.15, 0.25], [0.20, 0.30]], name='w1')
w2 = tf.Variable([[0.40, 0.50], [0.45, 0.55]], name='w2')
b1 = tf.Variable([0.35], name='b1')
b2 = tf.Variable([0.60], name='b2')

# Create model
def neural_network():
    layer_1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
    out_layer = tf.nn.sigmoid(tf.matmul(layer_1, w2) + b2)
    return out_layer, layer_1

# operations
V, hidden = neural_network()
error = tf.reduce_sum(0.5 * tf.square(Y - V))
grads = tf.gradients(error, tf.trainable_variables())
trainw1 = tf.assign_sub(w1, grads[0] * learning_rate)
trainw2 = tf.assign_sub(w2, grads[1] * learning_rate)
trainb1 = tf.assign_sub(b1, grads[2] * learning_rate)
trainb2 = tf.assign_sub(b2, grads[3] * learning_rate)

initial_data = [[0.05, 0.10]]
initial_labels = [[0.01, 0.99]]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print('h1, h2')
    print(sess.run(hidden, feed_dict={X: initial_data, Y: initial_labels}))
    print('o1, o2')
    print(sess.run(V, feed_dict={X: initial_data, Y: initial_labels}))
    print('w1, w2')
    print(sess.run([w1, w2]))
    print('error')
    print(sess.run(error, feed_dict={X: initial_data, Y: initial_labels}))
    print('gradients')
    print(sess.run(grads, feed_dict={X: initial_data, Y: initial_labels}))
    print('backprop')
    # print(sess.run([trainw1, trainw2, trainb1, trainb2], feed_dict={X: initial_data, Y: initial_labels}))
    # for i in range(10000):
    #     if i % 1000 == 0:
    #         print(i)
    sess.run([trainw1, trainw2], feed_dict={X: initial_data, Y: initial_labels})
    print('h1, h2')
    print(sess.run(hidden, feed_dict={X: initial_data, Y: initial_labels}))
    print('o1, o2')
    print(sess.run(V, feed_dict={X: initial_data, Y: initial_labels}))
    print('w1, w2')
    print(sess.run([w1, w2]))
    print('new error')
    print(sess.run(error, feed_dict={X: initial_data, Y: initial_labels}))
