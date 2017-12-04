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

        # conv = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, activation)
        # pool = tf.layers.max_pooling2d(inputs, pool_size, strides)
        conv1 = tf.layers.conv2d(self.tf_s, 32, 5, 1, 'same', activation=tf.nn.relu) # -> 80, 80, 32
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2) # -> 40, 40, 32 
        conv2 = tf.layers.conv2d(pool1, 64, 5, 1, activation=tf.nn.relu) # -> 36, 36, 64
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2) # -> 18, 18, 64 
        conv3 = tf.layers.conv2d(pool2, 64, 5, 1, activation=tf.nn.relu) # -> 14, 14, 64
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2) # -> 7, 7, 64 
        flat = tf.reshape(pool3, [-1, 7*7*64])
        fc1 = tf.layers.dense(flat, 256, tf.nn.relu)
        acts = tf.layers.dense(fc1, 10)

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
    
    def _preprocess(I):
        I = I[34:194]
        I = I[::2, ::2, 0] # 80x80
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()

    def train(self):
        episodes = 20000



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

