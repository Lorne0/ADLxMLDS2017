from agent_dir.agent import Agent

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.n_features = 6400
        self.n_actions = 6
        self.lr = 1e-4
        self.gamma = 0.99
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.tf_s = tf.placeholder(tf.float32, [None, self.n_features])
        self.tf_a = tf.placeholder(tf.int32, [None, ])
        self.tf_vt = tf.placeholder(tf.float32, [None, ])

        #fc1
        layer1 = tf.layers.dense(
            inputs = self.tf_s,
            units = 400,
            activation = tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
        )
        #fc2
        layer2 = tf.layers.dense(
            inputs = layer1,
            units = 200,
            activation = tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
        )
        #output
        acts = tf.layers.dense(
            inputs = layer2,
            units = self.n_actions,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
        )

        self.acts_prob = tf.nn.softmax(acts)
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=acts, labels=self.tf_a)
        loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()

