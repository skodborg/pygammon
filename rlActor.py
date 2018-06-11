import tensorflow as tf
import copy
import time
import benchmark
import randomWalk as rw
import random as rnd

class rlActor:
    def __init__(self, logspath='1', printStuff=False):
        tf.reset_default_graph()

        self.printStuff = printStuff

        self.sess = tf.Session()

        # network parameters
        self.n_hidden = 5
        self.n_input = 30
        self.n_output = 1

        # tf Graph input
        # shape [None, x] means any number of rows of x cols of data
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_input], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_output], name='Y')
        self.w1 = tf.Variable(tf.zeros([self.n_input, self.n_hidden]), name='w1')
        self.w2 = tf.Variable(tf.zeros([self.n_hidden, self.n_output]), name='w2')
        # self.w1 = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]), name='w1')
        # self.w2 = tf.Variable(tf.random_normal([self.n_hidden, self.n_output]), name='w2')

        # will hold the value omega = alpha * (y_tplus1 - y_t)
        self.omega = tf.placeholder(tf.float32, shape=[1,1], name='omega')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.incr_global_step_op = tf.assign_add(self.global_step, 1, name='incr_global_step_op')

        # Construct model
        self.model = self.singlelayer_perceptron(self.X)
        self.V = self.model

        # Initializing the variables
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

        # self.summary_writer = tf.summary.FileWriter('logs/%s' % VERSION, self.sess.graph)

    def print_w1(self):
        print(self.sess.run('w1:0'))


    def train(self, rounds=10000):

        alpha = 0.1  # learning rate for w1
        gamma = 0.7

        w2_update_op = tf.assign(self.w2, tf.add(self.w2, self.omega), name='w2_update')
        w1_update_op = tf.assign(self.w1, tf.add(self.w1, self.omega), name='w1_update')

        batch_sum_moves = 0
        batch_sum_reward = 0

        for i in range(rounds):
            self.sess.run(self.incr_global_step_op)

            walk = rw.RandomWalk()
            done, _, _, _ = walk.done()

            while not done:
                nn_repr = walk.nn_state_repr()
                y_t = self.sess.run(self.V, feed_dict={self.X: nn_repr})

                moves_results = []
                for move in walk.move_space():
                    walk_copy = rw.RandomWalk(move)
                    nn_repr_copy = walk_copy.nn_state_repr()
                    y_tplus1 = self.sess.run(self.V, feed_dict={self.X: nn_repr_copy})
                    moves_results.append((move, y_tplus1))

                best_move = max(moves_results, key=lambda x: x[1])[0]
                print(moves_results)
                if rnd.randint(0, 1) > 0.5 or moves_results[0][1] == moves_results[1][1]:
                    print('random decided')
                    best_move = rnd.choice(walk.move_space())
                print('best move: %s' % str(best_move))

                # perform it, reevaluate model in the new state
                walk.move(best_move)
                done, reward, state, _ = walk.done()
                print('STATE: %i' % state)
                new_nn_repr = walk.nn_state_repr()

                # if reward == 0:
                #     if done:
                #         reward = -5
                #     else:
                #         reward = -1
                # else:
                #     reward = 5

                # calculate Omega_k = y_tplus1 - y_t
                y_tplus1 = self.sess.run(self.model, feed_dict={self.X: new_nn_repr})
                omega = alpha * (reward + (gamma * y_tplus1) - y_t)
                print('omega = %f * (%i + (%f * %f) - %f) = %f' % (alpha, reward, gamma, y_tplus1, y_t, omega))

                tvars = tf.trainable_variables()
                cost = self.Y - self.V
                gradients = tf.gradients(cost, tvars)
                print(gradients)
                self.sess.run(tf.add(self.w1, gradients[0]))

                if state > 28:
                    print('BEFORE UPDATE')
                    weights1, weights2 = self.sess.run([self.w1, self.w2])
                    print('weights1', weights1)
                    print('weights2', weights2)
                self.sess.run([w1_update_op, w2_update_op], feed_dict={self.omega: omega})
                if state > 28:
                    weights1, weights2 = self.sess.run([self.w1, self.w2])
                    print('AFTER UPDATE')
                    print('weights1', weights1)
                    print('weights2', weights2)
                input()

            batch_sum_moves += walk.moves
            _, reward, _, _ = walk.done()
            print(reward)
            batch_sum_reward += reward

            if i % 100 == 0:
                print('round %i' % i)
                avg_moves = batch_sum_moves / 100
                avg_reward = batch_sum_reward / 100
                print('avg moves: %f' % avg_moves)
                print('avg reward: %f\n' % avg_reward)

                # reset
                batch_sum_moves = 0
                batch_sum_reward = 0


    # Create model
    def singlelayer_perceptron(self, x):
        # activation functions are key to let neural networks
        # fit to nonlinear functions! otherwise, they are just learning
        # a linear transformation, as all the weights an bias multiplication and addition
        # can essentially be reduced to a simple linear function (ax+b)

        # Hidden fully connected layer with 40 neurons
        # layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.w1), self.b1))
        layer_1 = tf.nn.sigmoid(tf.matmul(x, self.w1))
        # Output fully connected layer with a neuron for each class (here: 1)
        # out_layer = tf.nn.relu(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        out_layer = tf.matmul(layer_1, self.w2)

        return out_layer



def main():
    rl = rlActor()
    rl.train()

if __name__ == '__main__':
    main()
