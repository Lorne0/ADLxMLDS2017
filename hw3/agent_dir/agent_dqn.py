from agent_dir.agent import Agent
import random
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            #tf.train.Saver().restore(self.sess, "./model/dqn_model")

        self.env = env
        self.state_size = (84, 84, 4)
        self.action_size = self.env.action_space.n
        self.exploration_rate = 1.0
        self.exploration_delta = 9.5*1e-7 # after 1000000, exploration_rate will be 0.05
        self.lr = 1e-3
        self.gamma = 0.99
        self.batch_size = 32
        self.timestep = 0

        self.Memory = {}
        self.memory_limit = 10000
        self.memory_count = 0

        self.online_update_frequency = 4
        self.target_update_frequency = 1000
        self.build_model()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        #tf.train.Saver().restore(self.sess, "./model/dqn_model")

    def build_model(self):
        self.s = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.mask_target = tf.placeholder(tf.float32, [None, self.action_size])
        
        with tf.variable_scope('online'):
            self.online_net = self.build_net(self.s, 'online', [tf.GraphKeys.GLOBAL_VARIABLES, 'online_parameter'])
        with tf.variable_scope('target'):
            self.target_net = self.build_net(self.s_, 'target', [tf.GraphKeys.GLOBAL_VARIABLES, 'target_parameter'])

        self.loss = tf.reduce_mean(tf.squared_difference(self.mask_target, self.online_net))
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        online_parameter = tf.get_collection([tf.GraphKeys.GLOBAL_VARIABLES, 'online_parameter'])
        target_parameter = tf.get_collection([tf.GraphKeys.GLOBAL_VARIABLES, 'target_parameter'])
        self.update_target_op = [tf.assign(t, o) for t, o in zip(target_parameter, online_parameter)]

    def build_net(self, inputs, scope, collection_name):
        # online network
        with tf.variable_scope(scope):
            initializer = tf.contrib.keras.initializers.he_uniform()
            #collection_name = [scope+'_parameter', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = layers.convolution2d(inputs, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu, weights_initializer=initializer, variables_collections = collection_name)
            conv2 = layers.convolution2d(conv1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu, weights_initializer=initializer, variables_collections = collection_name)
            conv3 = layers.convolution2d(conv2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=initializer, variables_collections = collection_name)
            conv_out = layers.flatten(conv3)
            fc = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None, weights_initializer=initializer, variables_collections = collection_name)
            LeakyReLU = tf.contrib.keras.layers.LeakyReLU(alpha=0.3)
            fc_out = LeakyReLU(fc)
            output = layers.fully_connected(fc_out, num_outputs=self.action_size, activation_fn=None, weights_initializer=initializer, variables_collections = collection_name)
        return output

    def init_game_setting(self):
        pass

    def make_action(self, observation, test=True):
        epsilon = 0.05 if test==True else self.exploration_rate
        if random.random() < epsilon: # random choose action:
            return np.random.randint(0, self.action_size)
        else:
            actions = self.sess.run(self.online_net, feed_dict={self.s: np.expand_dims(observation, axis=0)})[0]
            vmax = max(actions)
            a = []
            for i in range(len(actions)):
                if actions[i]==vmax:
                    a.append(i)
            return random.choice(a)

    def memory_store(self, m):
        self.Memory[self.memory_count % self.memory_limit] = m
        self.memory_count += 1

    def learn(self):
        ids = np.random.choice(min(self.memory_count, self.memory_limit), size=self.batch_size)
        s = np.zeros((self.batch_size, 84, 84, 4))
        actions = np.zeros(self.batch_size, dtype=np.int)
        rewards = np.zeros(self.batch_size)
        s_ = np.zeros((self.batch_size, 84, 84, 4))
        for i in range(self.batch_size):
            s[i] = self.Memory[ids[i]][0]
            actions[i] = self.Memory[ids[i]][1]
            rewards[i] = self.Memory[ids[i]][2]
            s_[i] = self.Memory[ids[i]][3]
        
        online_net, target_net = self.sess.run([self.online_net, self.target_net], 
                                                feed_dict={self.s: s, self.s_: s_})
        mask_target = online_net.copy()
        mask_target[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(target_net, axis=1)
        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.s: s, self.mask_target: mask_target}) 

    def train(self):
        saver = tf.train.Saver()
        episodes = 10000000
        result = [] #result for each episode
        for e in range(1, episodes+1):
            obs = self.env.reset()
            episode_reward = 0
            while True:
                a = self.make_action(obs, test=False)
                obs_, r, done, info = self.env.step(a)
                episode_reward += r
                self.memory_store((obs, a, r, obs_))

                self.timestep += 1

                if self.timestep > self.memory_limit and (self.timestep%self.target_update_frequency)==0:
                    self.sess.run(self.update_target_op)

                if self.timestep > self.memory_limit and (self.timestep%self.online_update_frequency)==0:
                    self.learn()

                if self.exploration_rate >= 0.05:
                    self.exploration_rate -= self.exploration_delta

                obs = obs_

                if done:
                    result.append(episode_reward)
                    break

            rr = np.mean(result[-100:])
            print("Episode: %d | Reward: %d | Last 100: %f | timestep: %d | exploration: %f" %(e, episode_reward, rr, self.timestep, self.exploration_rate))
            if (e%10) == 0:
                np.save('./result/dqn_result3.npy',result)
                save_path = saver.save(self.sess, "./model/dqn_model3")



