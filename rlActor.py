import tensorflow as tf
import argparse
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
        rwalk = rw.RandomWalk()

        # network parameters
        self.n_hidden = 5
        self.n_input = len(rwalk.state_space())
        self.n_output = 1

        # tf Graph input
        # shape [None, x] means any number of rows of x cols of data
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_input], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_output], name='Y')
        self.w1 = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]), name='w1')
        self.w2 = tf.Variable(tf.random_normal([self.n_hidden, self.n_output]), name='w2')
        self.b1 = tf.Variable([1.0], name='b1')
        self.b2 = tf.Variable([1.0], name='b2')

        # Construct model
        self.V = self.neural_network()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.incr_global_step_op = tf.assign_add(self.global_step, 1, name='incr_global_step_op')

        # Initializing the variables
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)


    def print_w1(self):
        print(self.sess.run('w1:0'))


    '''
    TODO: 
        consider a randomWalk of length 3
        starting in the middle will yield a reward immediately
        after taking any action
        Make sure the gradients and weight update operations
        function as expected, i.e. increasing estimate when
        observing a positive reward, lessening the estimate
        when observing a negative reward
    '''
    def train(self, rounds=100000):
        alpha = 0.5  # learning rate for w1
        gamma = 0.7  # 0.7
        explore_rate = 0.5

        reward = tf.placeholder(tf.float32, shape=[1,1], name='reward')
        sign = tf.placeholder(tf.float32, shape=[1], name='sign')
        # cost = reward + (gamma * self.Y) - self.V
        target = reward + gamma * self.Y
        cost = reward + gamma * self.Y - self.V  # square a good idea? do we need the sign?
        gradients = tf.gradients(cost, tf.trainable_variables())

        w1_update_op = tf.assign_sub(self.w1, alpha * cost * gradients[0]) 
        w2_update_op = tf.assign_sub(self.w2, alpha * cost * gradients[1]) 
        b1_update_op = tf.assign_sub(self.b1, alpha * cost[0] * gradients[2]) 
        b2_update_op = tf.assign_sub(self.b2, alpha * cost[0] * gradients[3]) 

        VERSION = '1'
        self.summary_writer = tf.summary.FileWriter('logs/%s' % VERSION, self.sess.graph)
        # self.printStuff = True

        for i in range(rounds):
            print('\n\nround %i' % i)
            self.sess.run(self.incr_global_step_op)

            walk = rw.RandomWalk()
            done, _, _, _ = walk.done()
            j = 0

            if i % 10 == 0 and i > 0 and explore_rate > 0.0:
                explore_rate -= 0.1

            if explore_rate <= 0.01:
                self.printStuff = True

            while not done:
                j += 1
                if j == 100:
                    break
                # y_t = V(s_t) : the value estimate of current state
                nn_repr = walk.nn_state_repr()
                y_t = self.sess.run(self.V, feed_dict={self.X: nn_repr})

                # evaluate all possible moves by simulating and estimating them using V
                moves_results = []
                curr_state = walk.getState()
                for move in walk.move_space():
                    walk_copy = rw.RandomWalk(curr_state)
                    walk_copy.move(move)
                    nn_repr_copy = walk_copy.nn_state_repr()
                    y_tplus1 = self.sess.run(self.V, feed_dict={self.X: nn_repr_copy})
                    if self.printStuff:
                        print('considering future board %s getting estimate %f' % (str(nn_repr_copy), y_tplus1))
                    moves_results.append((move, y_tplus1))

                # perform best move (acc. to V on all possible moves resulting states)
                # y_tplus1 = V(s_t+1) : the value estimate of next state
                best_move_tup = max(moves_results, key=lambda x: x[1])
                best_move = best_move_tup[0]
                if self.printStuff:
                    print('chose best_move %s with estimate %s' % (str(best_move), str(best_move_tup[1])))
                if rnd.random() < explore_rate:
                    # ensure exploration 10% of the time
                    best_move = rnd.choice(moves_results)[0]
                    if self.printStuff:
                        print('EXPLORING: chose %s' % str(best_move))
                walk.move(best_move)
                done, reward, state, _ = walk.done()
                new_nn_repr = walk.nn_state_repr()
                y_tplus1 = self.sess.run(self.V, feed_dict={self.X: new_nn_repr})
                # TODO: consider if rewards above 1 makes sense? if this is sufficient to eliminate
                if reward == [[1]]:
                    y_tplus1 = [[0.0]]
                # assert y_tplus1 == best_move_tup[1]
                cost_sign = self.sess.run(cost, feed_dict={'reward:0': reward, self.Y: y_tplus1, self.X: nn_repr})[0]
                cost_sign = [-1] if cost_sign[0] < 0 else [1]

                if self.printStuff:
                    print('reward: %s' % str(reward))
                    print('BEFORE UPDATE')
                    weights1, weights2, bias1, bias2 = self.sess.run([self.w1, self.w2, self.b1, self.b2])
                    # print('weights1', weights1)
                    # print('weights2', weights2)

                    print('V(s_t): %s' % str(y_t))
                    print('V(s_t+1): %s' % str(y_tplus1))
                    print('target: %s' % str(self.sess.run(target, feed_dict={'reward:0': reward, self.Y: y_tplus1})))
                    print('cost: %s' % str(self.sess.run(cost, feed_dict={'reward:0': reward, self.Y: y_tplus1, self.X: nn_repr})))
                    # print('gradients: %s' % str(self.sess.run(gradients, feed_dict={'reward:0': reward, self.Y: y_tplus1, self.X: nn_repr})))

                # GRADIENT DESCENT
                # w2 before w1 as we need to propagate w2 gradient changes to w1
                self.sess.run([w2_update_op, b2_update_op], feed_dict={self.X: nn_repr, self.Y: y_tplus1, 'reward:0': reward, 'sign:0': cost_sign})
                self.sess.run([w1_update_op, b1_update_op], feed_dict={self.X: nn_repr, self.Y: y_tplus1, 'reward:0': reward, 'sign:0': cost_sign})

                if self.printStuff:
                    weights1, weights2, bias1, bias2 = self.sess.run([self.w1, self.w2, self.b1, self.b2])
                    print('AFTER UPDATE')
                    # print('weights1', weights1)
                    # print('weights2', weights2)
                    # print('bias1', bias1)
                    # print('bias2', bias2)
                    print('new V(s_t) post update: %s' % str(self.sess.run(self.V, feed_dict={self.X: nn_repr})))
                    print('target: %s' % str(self.sess.run(target, feed_dict={'reward:0': reward, self.Y: y_tplus1})))
                    print('cost: %s' % str(self.sess.run(cost, feed_dict={'reward:0': reward, self.Y: y_tplus1, self.X: nn_repr})))
                    input()

            _, reward, _, _ = walk.done()
            print(str(walk.moves) + ' ' + str(reward) + ': ' + str(reward[0][0] / j) + '  \t explore_rate: %f' % explore_rate)
            if self.printStuff:
                input()

            # tf.reset_default_graph()


    # Create model
    def neural_network(self):
        # TODO: add sigmoid?
        layer_1 = tf.sigmoid(tf.matmul(self.X, self.w1) + self.b1)
        out_layer = tf.sigmoid(tf.matmul(layer_1, self.w2) + self.b2)
        return out_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='prints stuff during execution')
    args = parser.parse_args()
    print(args.verbose)
    printStuff = True if args.verbose else False
    rl = rlActor(printStuff=printStuff)
    rl.train()

if __name__ == '__main__':
    main()
