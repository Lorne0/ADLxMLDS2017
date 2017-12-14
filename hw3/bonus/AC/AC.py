import numpy as np
import tensorflow as tf

np.random.seed(1229)
tf.set_random_seed(1229)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr, num_units):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.num_units = num_units
        self.s = tf.placeholder(tf.float32, [1, self.n_features])
        self.a = tf.placeholder(tf.int32, None)
        self.td_error = tf.placeholder(tf.float32, None)

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(self.s, self.num_units, tf.nn.relu)
            self.acts_prob = tf.layers.dense(l1, self.n_actions, tf.nn.softmax)

        self.exp_v = tf.reduce_mean(tf.log(self.acts_prob[0, self.a])*self.td_error)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        feed_dict = {self.s: s[np.newaxis, :], self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, {self.s: s[np.newaxis, :]})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self, sess, n_features, lr, num_units):
        self.sess = sess
        self.n_features = n_features
        self.lr = lr
        self.num_units = num_units
        self.gamma = 0.9
        self.s = tf.placeholder(tf.float32, [1,self.n_features])
        self.v_ = tf.placeholder(tf.float32, [1,1])
        self.r = tf.placeholder(tf.float32, None)

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(self.s, self.num_units, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1, None)

        self.td_error = self.r + self.gamma * self.v_ - self.v
        self.loss = tf.square(self.td_error)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error
            
