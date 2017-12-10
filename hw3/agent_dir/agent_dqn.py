from agent_dir.agent import Agent
import random
import scipy.misc
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.layers as layers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K


def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())


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
            self.online_model = load_model('./model/dqn_keras_online_model.h5')
            self.env = env
            self.action_size = self.env.action_space.n
            self.exploration_rate = 1.0
            #tf.train.Saver().restore(self.sess, "./model/dqn_model")
        else:
            self.env = env
            self.state_size = (84, 84, 4)
            self.action_size = self.env.action_space.n
            self.exploration_rate = 1.0
            self.exploration_delta = 9.5*1e-7 # after 1000000, exploration_rate will be 0.05
            #self.exploration_delta = 6.33333*1e-7 # after 1500000, exploration_rate will be 0.05
            self.lr = 1e-4
            self.gamma = 0.99
            self.batch_size = 32
            self.timestep = 0

            self.Memory = {}
            self.memory_limit = 10000
            self.memory_count = 0

            self.online_update_frequency = 4
            self.target_update_frequency = 1000
            #self.build_model()
            self.online_model = self.build_model()
            self.target_model = self.build_model()
            self.update_target_model()

            #self.optimizer = self.optimizer()
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            #K.set_session(self.sess)
            #self.sess = tf.Session()
            #self.sess.run(tf.global_variables_initializer())

            '''
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            '''
            
            #tf.train.Saver().restore(self.sess, "./model/dqn_model_sum")
    '''
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')
        py_x = self.online_model.output
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        opt = RMSprop(lr=self.lr, epsilon=0.01)
        updates = opt.get_updates(self.online_model.trainable_weights, [], loss)
        train = K.function([self.online_model.input, a, y], [loss], updates=updates)
        return train
    '''

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size, kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_uniform'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        #model.add(Dense(512, activation='linear'))
        #model.add(LeakyReLU())
        model.add(Dense(self.action_size, kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model
        '''
        self.s = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.mask_target = tf.placeholder(tf.float32, [None, self.action_size])
        
        with tf.variable_scope('online'):
            self.online_net = self.build_net(self.s, 'online', [tf.GraphKeys.GLOBAL_VARIABLES, 'online_parameter'])
        with tf.variable_scope('target'):
            self.target_net = self.build_net(self.s_, 'target', [tf.GraphKeys.GLOBAL_VARIABLES, 'target_parameter'])

        self.loss = tf.reduce_mean(tf.squared_difference(self.mask_target, self.online_net))
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        online_parameter = tf.get_collection('online_parameter')
        target_parameter = tf.get_collection('target_parameter')
        self.update_target_op = [tf.assign(t, o) for t, o in zip(target_parameter, online_parameter)]
        '''

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    '''
    def build_net(self, inputs, scope, collection_name):
        # online network
        with tf.variable_scope(scope):
            initializer = tf.contrib.keras.initializers.he_uniform()
            #collection_name = [scope+'_parameter', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = layers.convolution2d(inputs, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu, weights_initializer=initializer, variables_collections = collection_name)
            conv2 = layers.convolution2d(conv1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu, weights_initializer=initializer, variables_collections = collection_name)
            conv3 = layers.convolution2d(conv2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=initializer, variables_collections = collection_name)
            conv_out = layers.flatten(conv3)
            fc = layers.fully_connected(conv_out, num_outputs=512, activation_fn=tf.nn.relu, weights_initializer=initializer, variables_collections = collection_name)
            #LeakyReLU = tf.contrib.keras.layers.LeakyReLU(alpha=0.3)
            #fc_out = LeakyReLU(fc)
            output = layers.fully_connected(fc, num_outputs=self.action_size, activation_fn=None, weights_initializer=initializer, variables_collections = collection_name)
        return output
    '''

    def init_game_setting(self):
        pass

    def make_action(self, observation, test=True):
        epsilon = 0.0005 if test==True else self.exploration_rate
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_size)
        else:
            q = self.online_model.predict(np.expand_dims(observation, axis=0))
            return np.argmax(q[0])
        '''
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
        '''

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
        
        q_online = self.online_model.predict(s)
        q_target = self.target_model.predict(s_)
        y = q_online.copy()
        y[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(q_target, axis=1)
        self.online_model.train_on_batch(s, y)
        #q_target = self.target_model.predict(s_)
        #y = rewards + self.gamma * np.max(q_target, axis=1)
        #loss = self.optimizer([s, actions, y])


        '''
        online_net, target_net = self.sess.run([self.online_net, self.target_net], 
                                                feed_dict={self.s: s, self.s_: s_})
        mask_target = online_net.copy()
        mask_target[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(target_net, axis=1)
        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.s: s, self.mask_target: mask_target}) 
        '''

    def train(self):
        #saver = tf.train.Saver()
        episodes = 1000000
        result = [] #result for each episode
        for e in range(1, episodes+1):
            obs = self.env.reset()
            episode_reward = 0
            while True:
                a = self.make_action(obs, test=False)
                obs_, r, done, info = self.env.step(a)
                episode_reward += r
                self.memory_store((obs.copy(), a, r, obs_.copy()))

                self.timestep += 1

                if self.timestep > self.memory_limit and (self.timestep%self.target_update_frequency)==0:
                    self.update_target_model()
                    #self.sess.run(self.update_target_op)

                if self.timestep > self.memory_limit and (self.timestep%self.online_update_frequency)==0:
                    self.learn()

                if self.exploration_rate > 0.05:
                    self.exploration_rate -= self.exploration_delta

                obs = obs_.copy()

                if done:
                    result.append(episode_reward)
                    break

            rr = np.mean(result[-100:])
            print("Episode: %d | Reward: %d | Last 100: %f | timestep: %d | exploration: %f" %(e, episode_reward, rr, self.timestep, self.exploration_rate))
            if (e%10) == 0:
                np.save('./result/dqn_keras_result.npy',result)
                self.online_model.save('./model/dqn_keras_online_model.h5')
                self.target_model.save('./model/dqn_keras_target_model.h5')
                #save_path = saver.save(self.sess, "./model/dqn_model03")



