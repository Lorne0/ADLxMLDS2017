import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import gym

np.random.seed(1229)
tf.set_random_seed(1229)
#num_workers = multiprocessing.cpu_count()
num_workers = 6
GLOBAL_NET_SCOPE = "global_net"
GLOBAL_EPISODE = 0 # max: 1000
GLOBAL_RESULT = []
gamma = 0.9
entropy_beta = 0.001
lr_a = 0.001
lr_c = 0.001

env = gym.make('CartPole-v0').unwrapped
n_features = env.observation_space.shape[0]
n_actions = env.action_space.n

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_features])
                self.a_params, self.c_params = self.build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_features])
                self.a = tf.placeholder(tf.int32, [None, 1])
                self.v_target = tf.placeholder(tf.float32, [None, 1])
                self.a_prob, self.v, self.a_params, self.c_params = self.build_net(scope)
                
                td_error = tf.subtract(self.v_target, self.v)
                self.c_loss = tf.reduce_mean(tf.square(td_error))

                log_prob = tf.reduce_sum(tf.log(self.a_prob)*tf.one_hot(self.a, n_actions, dtype=tf.float32), axis=1, keep_dims=True)
                exp_v = log_prob * td_error
                entropy = -tf.reduce_sum(self.a_prob*tf.log(self.a_prob+1e-6), axis=1, keep_dims=True)
                self.exp_v = entropy_beta * entropy + exp_v # exploration
                self.a_loss = tf.reduce_mean(-self.exp_v)

                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'): # global -> local
                    self.pull_a_params_op = [l.assign(g) for l,g in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l.assign(g) for l,g in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'): # local -> global
                    self.push_a_params_op = opt_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.push_c_params_op = opt_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def build_net(self, scope):
        w_init = tf.random_normal_initializer(0.0, 0.1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu, kernel_initializer=w_init)
            a_prob = tf.layers.dense(l_a, n_actions, tf.nn.softmax, kernel_initializer=w_init)
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu, kernel_initializer=w_init)
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')

        return a_prob, v, a_params, c_params

    def push_global(self, ss, aa, R):
        sess.run([self.push_a_params_op, self.push_c_params_op], {self.s: ss, self.a: aa, self.v_target: R})

    def pull_global(self):
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        probs = sess.run(self.a_prob, {self.s: np.expand_dims(s, axis=0)})
        a = np.random.choice(range(probs.shape[1]), p=probs.ravel())
        return a

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make('CartPole-v0').unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_EPISODE, GLOBAL_RESULT
        ss, aa, rr = [], [], []
        while not COORD.should_stop() and GLOBAL_EPISODE <= 1000:
            s = self.env.reset()
            ep_r = 0
            t = 0
            while True:
                a = self.AC.choose_action(s)
                s_, true_r, done, info = self.env.step(a)
                ep_r += true_r
                t += 1

                x, x_dot, theta, theta_dot = s_
                r1 = (self.env.x_threshold - abs(x) ) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                r = r1 + r2

                ss.append(s)
                aa.append(a)
                rr.append(r)

                if t >= 40000: done=True

                if done or t%10==0:
                    ##### on paper #####
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = sess.run(self.AC.v, {self.AC.s: np.expand_dims(s_, axis=0)})[0][0]

                    R = []
                    for r in rr[::-1]:
                        v_s_ = r + gamma * v_s_
                        R.append(v_s_)
                    R.reverse()

                    self.AC.push_global(np.vstack(ss), np.vstack(aa), np.vstack(R))
                    ss, aa, rr = [], [], []
                    self.AC.pull_global()
                    ###################

                s = s_
                if done:
                    GLOBAL_RESULT.append(ep_r)
                    r30 = np.mean(GLOBAL_RESULT[-30:])
                    print("Episode: %d | Reward: %d | Last 30 Reward: %f" %(GLOBAL_EPISODE, ep_r, r30))
                    GLOBAL_EPISODE += 1
                    break

if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    config = tf.ConfigProto(gpu_options = gpu_options)
    sess = tf.Session(config=config)

    with tf.device('/cpu:0'):
        opt_A = tf.train.RMSPropOptimizer(lr_a)
        opt_C = tf.train.RMSPropOptimizer(lr_c)
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []
        for i in range(num_workers):
            i_name = "W_" + str(i)
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    print("End of training A3C")

np.save('../result/Cartpole_A3C_result.npy', GLOBAL_RESULT)
