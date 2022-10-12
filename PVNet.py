
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
class PVNet2():
    def __init__(self, board_height, board_width):
        
        self.board_width = board_width
        self.board_height = board_height

        # Define the tensorflow neural network
        # 1. Input:
        tf.reset_default_graph()
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 4, board_height, board_width])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        # 2. Common Networks Layers
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            data_format="channels_last",
                                            activation=tf.nn.relu)
        # Flatten the tensor
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)
        self.saver = tf.train.Saver()
        
        '''
        self.board_width = board_width
        self.board_height = board_height

        tf.reset_default_graph()
        # Define the tensorflow neural network
        # 1. Input:
        #tf.reset_default_graph()
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 4, board_height, board_width])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        
        # Conv layer followed by relu activation and batchnormalization
        self.x = tf.layers.conv2d(inputs=self.input_state, filters=64, kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
        self.x = tf.layers.batch_normalization(inputs=self.x)
        self.x = tf.layers.conv2d(inputs=self.x, filters=128,kernel_size=[3, 3], padding="same",activation=tf.nn.relu)
        self.x = tf.layers.batch_normalization(inputs=self.x)
        self.x = tf.layers.conv2d(inputs=self.x, filters=256,kernel_size=[3, 3], padding="same",activation=tf.nn.relu)
        self.x = tf.layers.batch_normalization(inputs=self.x)
        #print(self.x.shape)
        # Policy Net
        self.p = tf.layers.conv2d(inputs=self.x, filters=2,kernel_size=[1, 1], padding="same",activation=tf.nn.relu)
        self.p = tf.layers.batch_normalization(inputs=self.p)
        #print(self.p.shape)
        self.p = tf.reshape(self.p, [-1, 2 * board_height * board_width])
        #print(self.p.shape)
        self.action_fc = tf.layers.dense(inputs=self.p, units=board_height * board_width, activation=tf.nn.log_softmax)
        
        # Value net
        self.v = tf.layers.conv2d(inputs=self.x, filters=1, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
        self.v = tf.layers.batch_normalization(inputs=self.v)
        self.v = tf.reshape(self.v, [-1,  board_height * board_width])
        self.v = tf.layers.dense(inputs=self.v, units=64, activation=tf.nn.relu)
        self.evaluation_fc = tf.layers.dense(inputs=self.v, units=1, activation=tf.nn.tanh)

        
        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        '''
        
    def predict(self, state, v=True):
        #state.get_state().reshape(-1, 1, 8, 8)
        log_act_probs, value = self.session.run(
            [self.action_fc, self.evaluation_fc2],
            feed_dict={self.input_states: state}
            )
        act_probs = np.exp(log_act_probs)
        #print("predict", act_probs, value[0][0])
        if v:
            return act_probs[0], value[0][0]
        else:
            return act_probs, value
       
    
    def train(self, state_batch, mcts_probs, winner_batch, lr):
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy
    
    def save(self, path):
        self.saver.save(self.session, path)
        
    def restore(self, path):
        self.saver.restore(self.session, path)