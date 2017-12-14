import numpy as np
import tensorflow as tf
import random, sys
import gym

np.random.seed(1229)
tf.set_random_seed(1229)

class Dueling_DQN:
    def __init__(self, sess, n_features, n_actions, memory_limit, replace_step, batch_size, lr, epsilon_max, epsilon_increment, gamma, num_units):
        self.n_features = n_features
        self.n_actions = n_actions
        self.memory_limit = memory_limit #2000
        self.memory_count = 0
        #self.Memory = np.zeros((self.memory_limit, 2*self.n_features+2)) # s: n_features, s_: n_features, a: 1, r: 1
        self.Memory = {} # tuple (s, a, r, s_, done)

        self.replace_step = replace_step #100
        self.batch_size = batch_size #8
        self.lr = lr #0.01
        self.epsilon_max = epsilon_max #0.95
        self.epsilon_increment = epsilon_increment #0.001
        self.epsilon = 0
        self.gamma = gamma #0.9
        self.learn_step = 0

        self.num_units = num_units #10

        self._initialize_net()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer()) 


    def _initialize_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')
        w_initial = tf.random_normal_initializer(0.0, 0.3)
        b_initial = tf.constant_initializer(0.1)

        # evaluate net
        with tf.variable_scope('evaluate'):
            collection_name = ['evaluate_parameter', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.num_units], initializer=w_initial, collections=collection_name)
                b1 = tf.get_variable('b1', [1, self.num_units], initializer=b_initial, collections=collection_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [self.num_units, 1], initializer=w_initial, collections=collection_name)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initial, collections=collection_name)
                self.V = tf.matmul(l1, w2)+b2
            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [self.num_units, self.n_actions], initializer=w_initial, collections=collection_name)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initial, collections=collection_name)
                self.A = tf.matmul(l1, w2) + b2
            with tf.variable_scope("Q"):
                self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr, decay=0.99).minimize(self.loss)

        # target net 
        with tf.variable_scope('target'):
            collection_name = ['target_parameter', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.num_units], initializer=w_initial, collections=collection_name)
                b1 = tf.get_variable('b1', [1, self.num_units], initializer=b_initial, collections=collection_name)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [self.num_units, 1], initializer=w_initial, collections=collection_name)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initial, collections=collection_name)
                self.V = tf.matmul(l1, w2)+b2
            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [self.num_units, self.n_actions], initializer=w_initial, collections=collection_name)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initial, collections=collection_name)
                self.A = tf.matmul(l1, w2) + b2
            with tf.variable_scope("Q"):
                self.q_next = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

    def epsilon_greedy(self, obs):
        # ex: actions = [0.1, 0.07, 0.1, 0.05]
        # random choose if the max indexes are more than one
        obs = obs[np.newaxis, :]
        if random.random() > self.epsilon: # random choose action:
            return np.random.randint(0, self.n_actions)
        else: # choose the best action
            actions = self.sess.run(self.q_eval, feed_dict={self.s: obs})[0]
            vmax = max(actions)
            a = []
            for i in range(len(actions)):
                if actions[i]==vmax:
                    a.append(i)
            return random.choice(a)

    def memory_store(self, m):
        self.Memory[self.memory_count % self.memory_limit] = m
        self.memory_count += 1

    def replace_target(self):
        tp = tf.get_collection('target_parameter')
        ep = tf.get_collection('evaluate_parameter')
        self.sess.run([tf.assign(t, e) for t, e in zip(tp, ep)])

    def learn(self):
        self.learn_step += 1
        if self.learn_step % self.replace_step == 0:
            self.replace_target()

        ids = np.random.choice(min(self.memory_count, self.memory_limit), size=self.batch_size)
        s = np.zeros((self.batch_size, self.n_features))
        a = np.zeros(self.batch_size, dtype=np.int)
        r = np.zeros(self.batch_size)
        s_ = np.zeros((self.batch_size, self.n_features))
        done = np.zeros(self.batch_size, dtype=np.int)
        for i in range(self.batch_size):
            s[i] = self.Memory[ids[i]][0]
            a[i] = self.Memory[ids[i]][1]
            r[i] = self.Memory[ids[i]][2]
            s_[i] = self.Memory[ids[i]][3]
            done[i] = self.Memory[ids[i]][4]

        q_eval, q_next = self.sess.run([self.q_eval, self.q_next], feed_dict={self.s: s, self.s_: s_})
        q_target = q_eval.copy()
        for i in range(self.batch_size):
            if done[i]==1:
                q_target[i, a[i]] = r[i]
            else:
                q_target[i, a[i]] = r[i] + self.gamma * np.max(q_next, axis=1)[i]

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.s: s, self.q_target:q_target})

        self.epsilon = self.epsilon+self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max
