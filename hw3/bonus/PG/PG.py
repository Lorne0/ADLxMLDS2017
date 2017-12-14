import numpy as np
import tensorflow as tf
import gym

np.random.seed(1229)
tf.set_random_seed(1229)

class PolicyGradient:
    def __init__(self, sess, n_features, n_actions, lr, gamma, num_units):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr #0.02
        self.gamma = gamma #0.99 # reward decay
        self.num_units = num_units
        self._build_net()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_s = tf.placeholder(tf.float32, [None, self.n_features], name='obs')
            self.tf_a = tf.placeholder(tf.int32, [None, ], name='act')
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name='vt')

        # fc1
        layer1 = tf.layers.dense(
            inputs = self.tf_s,
            units = self.num_units,
            activation = tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1), 
            name='fc1'
        )
        
        # fc2
        acts = tf.layers.dense(
            inputs = layer1,
            units = self.n_actions,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1), 
            name='fc2'
        )

        self.acts_prob = tf.nn.softmax(acts, name='acts_prob')
        with tf.name_scope('loss'):
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=acts, labels=self.tf_a)
            neg_log_prob = tf.reduce_sum(-tf.log(self.acts_prob)*tf.one_hot(self.tf_a, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, obs):
        prob = self.sess.run(self.acts_prob, feed_dict={self.tf_s: obs[np.newaxis, :]})
        return np.random.choice(range(prob.shape[1]), p=prob.ravel()) # random
        #return np.argmax(prob.ravel()) # deterministic

    def _discounted_reward(self, r):
        dr = np.zeros_like(r)
        a = 0
        for t in reversed(range(len(r))):
            a = a * self.gamma + r[t]
            dr[t] = a
        dr -= np.mean(dr)
        dr /= np.std(dr)
        return dr
        
    def learn(self, ss, aa, rr):
        dr = self._discounted_reward(rr)
        self.sess.run(self.train, feed_dict={
            self.tf_s: np.vstack(ss), 
            self.tf_a: np.array(aa),
            self.tf_vt: dr
        })
        return dr
        
