import tensorflow as tf
import copy
import game
import time

VERSION = 'v4'

'''
new version

- learning_rate: 0.001 --> 0.1
- neural network layers: 198 x 80 x 40 x 1  -->  198 x 40 x 1

'''
class v4_nnActor:
    def __init__(self, logspath=VERSION, printStuff=False):
        tf.reset_default_graph()

        self.printStuff = printStuff

        self.sess = tf.Session()

        # network parameters
        self.n_hidden = 40
        self.n_input = 198
        self.n_output = 1


        # tf Graph input
        # shape [None, x] means any number of rows of x cols of data
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_input], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_output], name='Y')
        self.w1 = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]), name='w1')
        self.w2 = tf.Variable(tf.random_normal([self.n_hidden, self.n_output]), name='w2')

        # will hold the value omega = alpha * (y_tplus1 - y_t)
        self.omega = tf.placeholder(tf.float32, shape=[1,1], name='omega')

        # leave bias out for now?
        # self.b1 = tf.Variable(tf.random_normal([self.n_hidden]), name='b1')
        # self.b2 = tf.Variable(tf.random_normal([self.n_output]), name='b2')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.incr_global_step_op = tf.assign_add(self.global_step, 1, name='incr_global_step_op')

        # Construct model
        self.model, self.hiddenlayer = self.singlelayer_perceptron(self.X)

        # Initializing the variables
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

        # Initializing ops to save and restore all variables
        self.saver = tf.train.Saver(max_to_keep=3)
        self.steps_per_save = 1000

        self.summary_writer = tf.summary.FileWriter('actors/logs/%s' % VERSION, self.sess.graph)

    def print_w1(self):
        print(self.sess.run('w1:0'))

    def restore(self):
        model_to_load = tf.train.latest_checkpoint('./model/%s/' % VERSION) + '.meta'
        self.saver = tf.train.import_meta_graph(model_to_load)
        print('loaded meta graph %s' % model_to_load)

        self.saver.restore(self.sess, tf.train.latest_checkpoint('./model/%s/' % VERSION))
        print('restored variables')

        # print('w1 Tensor content:')
        # print(self.sess.run('w1:0'))

    def train(self, rounds=10000):

        # TODO: use decaying learning rates
        # alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step, \
            # 40000, 0.96, staircase=True), name='alpha')
        alpha = 0.1  # learning rate for w1
        beta = 0.1   # learning rate for w2

        # used for eligibility traces, not strictly necessary
        # delta = 0.7  # decay parameter
        # omega = lambda yk_t, yk_tplus1: yk_tplus1 - yk_t

        w2_update_op = tf.assign(self.w2, tf.add(self.w2, self.omega), name='w2_update')
        w1_update_op = tf.assign(self.w1, tf.add(self.w1, self.omega), name='w1_update')

        for i in range(rounds):
            # save model at end of the round if it has been too long since last
            if i % self.steps_per_save == 0 and i > 0:
                save_path = self.saver.save(self.sess, 'model/%s/198x40x1' % VERSION, global_step=self.global_step)
                print('saved model in path: %s' % save_path)

            self.sess.run(self.incr_global_step_op)
            gamestate = game.initialize_game()
            moves_count = 0

            while not game.find_winner(gamestate):
                moves_count += 1
                valid_moves = []
                curr_player = None
                start = time.time()
                if gamestate[2] == 'white':
                    valid_moves = game.white_valid_moves(gamestate)
                    curr_player = 'white'
                else:
                    valid_moves = game.black_valid_moves(gamestate)
                    curr_player = 'black'

                if not valid_moves:
                    gamestate = game.end_current_turn(gamestate)
                    continue

                nn_repr = game.nn_game_representation(gamestate)
                y_t = self.sess.run(self.model, feed_dict={self.X: nn_repr})

                # find best move
                # TODO: can become much prettier by saving states in a list
                #       and feed them all to the model, argmax'ing the result

                moves_results = []
                for move in valid_moves:
                    gamestate_copy = copy.deepcopy(gamestate)
                    from_pos = move[0]
                    to_pos = move[1]

                    gamestate_copy = game.move_piece(from_pos, to_pos, gamestate_copy)
                    moves_results.append((from_pos, to_pos, self.predict(game.nn_game_representation(gamestate_copy))))

                best_move = (None, None, -1)
                for move in moves_results:
                    if move[2] > best_move[2]:
                        best_move = move

                this_move = best_move

                # perform it, reevaluate model in the new state
                new_gamestate = game.move_piece(this_move[0], this_move[1], gamestate)
                new_nn_repr = game.nn_game_representation(new_gamestate)

                # calculate Omega_k = y_tplus1 - y_t
                  # y_tplus1 is the model value if the state is
                  # not terminal! If terminal, it is 1
                start = time.time()
                if game.find_winner(new_gamestate):
                    # a winner was just found after this_move, we must have won
                    #  - we know for sure the label of this data point should
                    #    be 1.0 then, no need to estimate
                    y_tplus1 = 1.0
                else:
                    # use Q state value for state t+1 as label for training t
                    y_tplus1 = self.sess.run(self.model, feed_dict={self.X: new_nn_repr})

                    # if it was the players last move, y_tplus1 is actually
                    # the probability that the opponent will win. Thus, the
                    # value we're looking for is 1.0 - y_tplus1
                    if new_gamestate[2] != curr_player:
                        y_tplus1 = 1.0 - y_tplus1

                omega = alpha * (y_tplus1 - y_t)
                # omega = tf.reduce_sum(y_tplus1 - y_t)

                # TODO: REFACTOR: use placeholders, define operation ONCE, reuse in sess.run
                # TODO2: try utilizing the below two variables, tvars and gradients
                # start = time.time()
                # tvars = tf.trainable_variables()
                # cost = self.model
                # gradients = tf.gradients(cost, tvars)
                # end = time.time()
                # print('5: ' + str(end - start))

                # update weights
                self.sess.run([w1_update_op, w2_update_op], feed_dict={self.omega: omega})

                # TODO: biases?? same rule?


                if game.find_winner(new_gamestate):
                    # continue with next round if a winner has been found
                    print('found winner! %s' % game.find_winner(new_gamestate))
                    print('total moves: %i\n' % moves_count)
                    break

                # update new state to be current state
                gamestate = new_gamestate



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
        out_layer = tf.nn.sigmoid(tf.matmul(layer_1, self.w2))

        return out_layer, layer_1


    def predict(self, x):
        sess = self.sess
        cost = sess.run(self.model, feed_dict={self.X: x})
        return cost


    def act(self, gamestate, valid_moves):
        assert len(valid_moves) > 0
        move_predictions = [(valid_moves[0], -1)]  # tuples of (move, prediction)
        for move in valid_moves:
            gamestate_copy = copy.deepcopy(gamestate)
            from_pos = move[0]
            to_pos = move[1]

            gamestate_copy = game.move_piece(from_pos, to_pos, gamestate_copy)
            nn_gamerepresentation = game.nn_game_representation(gamestate_copy)
            prediction = self.predict(nn_gamerepresentation)
            move_predictions.append((move, prediction))
        move = max(move_predictions, key=lambda x: x[1])[0]
        if self.printStuff:
            print('ai rolled %s' % str(gamestate[3]))
            print(move_predictions)
            print('moved %s' % str(move))
        return move


def main():
    actor = v4_nnActor()

if __name__ == '__main__':
    main()
