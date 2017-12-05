from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import tensorflow as tf

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
        #self.n_features = 6400
        self.env = env
        self.n_actions = 6
        self.lr = 1e-4
        self.gamma = 0.99
        self.prev_obs = None
        self._build_net()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        tf.train.Saver().restore(self.sess, "./model/pg_model")

    def _build_net(self):
        self.tf_s = tf.placeholder(tf.float32, [None, 80, 80, 1])
        self.tf_a = tf.placeholder(tf.int32, [None, ])
        self.tf_vt = tf.placeholder(tf.float32, [None, ])

        # conv = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, activation)
        # pool = tf.layers.max_pooling2d(inputs, pool_size, strides)
        # input 80*80
        conv1 = tf.layers.conv2d(self.tf_s, 16, 5, 2, activation=tf.nn.relu) # -> 38, 38, 16
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2) # -> 19, 19, 16 
        conv2 = tf.layers.conv2d(pool1, 32, 4, 1, activation=tf.nn.relu) # -> 16, 16, 32
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2) # -> 8, 8, 32 
        flat = tf.reshape(pool2, [-1, 8*8*32])
        #fc1 = tf.layers.dense(flat, 256, tf.nn.relu)
        acts = tf.layers.dense(flat, self.n_actions)

        self.acts_prob = tf.nn.softmax(acts)
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=acts, labels=self.tf_a)
        #loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
        loss = tf.reduce_sum(neg_log_prob * self.tf_vt)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_obs = None
    
    def preprocess(self, a):
        b = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
        b = b.astype(np.uint8)
        resized = scipy.misc.imresize(b, [80,80])
        return np.expand_dims(resized.astype(np.float32),axis=2)
        '''
        I = I[34:194]
        I = I[::2, ::2, 0] # 80x80
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()
        '''

    def train(self):
        saver = tf.train.Saver()
        result = []
        episodes = 20000
        for e in range(episodes):
            print("%d/%d" %(e+1, episodes))
            obs = self.env.reset()
            episode_reward = 0
            ss, aa, rr = [], [], []
            self.prev_obs = None
            while True:

                cur_obs = self.preprocess(obs)
                s = cur_obs - self.prev_obs if self.prev_obs is not None else np.zeros((80,80,1))
                self.prev_obs = cur_obs
                    
                a = self.make_action(s, test=False)
                obs_, r, done, info = self.env.step(a)
                episode_reward += r

                ss.append(s)
                aa.append(a)
                rr.append(r)

                if done:
                    vt = self.learn(ss,aa,rr)
                    result.append(episode_reward)
                    print("Reward: ", episode_reward)
                    break
                obs = obs_
            
            if (e+1)%10 == 0:
                save_path = saver.save(self.sess, "./model/pg_model")
                np.save("./result/pg.npy", result)
                
            rr = np.mean(result[-30:])
            print("Last 30 average reward: %f" %(rr))

        self.env.close()
        save_path = saver.save(self.sess, "./model/pg_model")
        np.save("./result/pg.npy", result)
                
    def make_action(self, observation, test=True):
        if test==False:
            prob = self.sess.run(self.acts_prob, feed_dict={self.tf_s:np.expand_dims(observation,axis=0)})
            return np.random.choice(range(prob.shape[1]), p=prob.ravel())
        else:
            cur_obs = preprocess(observation)
            s = cur_obs - self.prev_obs if self.prev_obs is not None else np.zeros((80,80,1))
            self.prev_obs = cur_obs
            prob = self.sess.run(self.acts_prob, feed_dict={self.tf_s:np.expand_dims(s,axis=0)})
            return np.random.choice(range(prob.shape[1]), p=prob.ravel())
    
    def discounted_reward(self, r):
        dr = np.zeros_like(r)
        a = 0
        for t in reversed(range(len(r))):
            a = a * self.gamma + r[t]
            dr[t] = a
        dr -= np.mean(dr)
        dr /= np.std(dr)
        return dr

    def learn(self, ss, aa, rr):
        dr = self.discounted_reward(rr)
        self.sess.run(self.train_op, feed_dict={self.tf_s:np.array(ss), self.tf_a:np.array(aa), self.tf_vt:dr})
        return dr

