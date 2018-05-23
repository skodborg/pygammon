import tensorflow as tf
import copy
import game

class nnActor:
    def __init__(self, logspath='1'):
        tf.reset_default_graph()

        self.sess = tf.Session()



        self.explore = 0.1
        self.learning_rate = 0.001

        # network parameters
        self.n_hidden_1 = 80
        self.n_hidden_2 = 40
        self.n_input = 198
        self.n_output = 1

        # tf Graph input
        # shape [None, x] means any number of rows of x cols of data
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_input], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_output], name='Y')


        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='h1'),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='h2'),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_output]), name='wout')
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name='b1'),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name='b2'),
            'out': tf.Variable(tf.random_normal([self.n_output]), name='bout')
        }


        # Create counter to keep track of steps (training iterations)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Construct model
        self.model = self.multilayer_perceptron(self.X)

        # Define loss and optimizer
        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        # self.loss_op = tf.reduce_sum(tf.square(self.Y - self.logits))
        self.loss_op = tf.reduce_sum(self.model)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step)

        # Initializing the variables
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

        # Initializing ops to save and restore all variables
        self.saver = tf.train.Saver(max_to_keep=5)
        self.steps_per_save = 100
        self.steps_left_until_save = self.steps_per_save

        # self.saver.restore(self.sess, tf.train.latest_checkpoint('./model/'))
        # print(self.sess.run('h1:0'))
        # print(self.sess.run('global_step:0'))

        self.summary_writer = tf.summary.FileWriter('actors/logs/%s' % logspath, self.sess.graph)


    # Create model
    def multilayer_perceptron(self, x):
        # activation functions are key to let neural networks
        # fit to nonlinear functions! otherwise, they are just learning
        # a linear transformation, as all the weights an bias multiplication and addition
        # can essentially be reduced to a simple linear function (ax+b)

        # Hidden fully connected layer with 80 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
        # Hidden fully connected layer with 40 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        # Output fully connected layer with a neuron for each class (here: 1)
        # TODO: this correct????
        out_layer = tf.nn.relu(tf.matmul(layer_2, self.weights['out']) + self.biases['out'])

        return out_layer


    def train(self, x, y):
        sess = self.sess
        # sess.run(self.init_op)

        # Run optimization op (backprop) and cost op (to get loss value)
        self.sess.run(self.train_op, feed_dict={self.X: x, self.Y: y})

        self.steps_left_until_save -= 1

        if self.steps_left_until_save <= 0:
            self.steps_left_until_save = self.steps_per_save
            save_path = self.saver.save(self.sess, 'model/v1', global_step=self.global_step)
            print('saved model in path: %s' % save_path)


    def predict(self, x):
        sess = self.sess
        # sess.run(self.init_op)
        cost = sess.run(self.loss_op, feed_dict={self.X: x})
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
        print(move_predictions)
        return max(move_predictions, key=lambda x: x[1])[0]


def main():
    actor = nnActor()

if __name__ == '__main__':
    main()