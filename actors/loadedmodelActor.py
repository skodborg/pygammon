import tensorflow as tf
import copy
import game

class loadedmodelActor:
    def __init__(self):
        tf.reset_default_graph()

        self.sess = tf.Session()

        self.n_input = 198
        self.n_output = 1


        # Initializing ops to save and restore all variables
        # self.saver = tf.train.Saver()
        model_to_load = tf.train.latest_checkpoint('./model/') + '.meta'
        self.saver = tf.train.import_meta_graph(model_to_load)
        print('loaded meta graph %s' % model_to_load)

        self.saver.restore(self.sess, tf.train.latest_checkpoint('./model/'))
        print('restored variables')


        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data
         
        self.graph = tf.get_default_graph()

         
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_input], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_output], name='Y')


        self.weights = {
            'h1': self.graph.get_tensor_by_name('h1:0'),
            'h2': self.graph.get_tensor_by_name('h2:0'),
            'out': self.graph.get_tensor_by_name('wout:0')
        }
        self.biases = {
            'b1': self.graph.get_tensor_by_name('b1:0'),
            'b2': self.graph.get_tensor_by_name('b2:0'),
            'out': self.graph.get_tensor_by_name('bout:0')
        }

        # Construct model
        self.model = self.multilayer_perceptron(self.X)

        # Define loss and optimizer
        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        # self.loss_op = tf.reduce_sum(tf.square(self.Y - self.logits))
        self.loss_op = tf.reduce_sum(self.model)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step)


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
        out_layer = tf.nn.relu(tf.matmul(layer_2, self.weights['out']) + self.biases['out'])

        return out_layer

    def predict(self, x):
        cost = self.sess.run(self.loss_op, feed_dict={self.X: x})
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
        print('bot performing move:')
        print(max(move_predictions, key=lambda x: x[1])[0])
        return max(move_predictions, key=lambda x: x[1])[0]


def main():
    actor = loadedmodelActor()

if __name__ == '__main__':
    main()
