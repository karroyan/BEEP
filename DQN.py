"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/
Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
gym: 0.8.1
"""
import tensorflow.compat.v1 as tf
import numpy as np
import config



class DQN:
    '''
    Class used to implement DQN
    '''
    def __init__(self, N_state, N_action):

        np.random.seed(50)
        tf.set_random_seed(50)
        tf.disable_eager_execution()

        self.name = 'DQN'
        self.N_state = N_state
        self.N_action = N_action
        self.memory_capacity = config.MEMORY_CAPACITY
        self.memory = np.zeros((self.memory_capacity, N_state * 2 + 2))
        self.MEMORY_COUNTER = 0
        self.LEARNING_STEP_COUNTER = 0

        # tf placeholders
        self.tf_s = tf.placeholder(tf.float32, [None, self.N_state])
        self.tf_a = tf.placeholder(tf.int32, [None, ])
        self.tf_r = tf.placeholder(tf.float32, [None, ])
        self.tf_s_ = tf.placeholder(tf.float32, [None, self.N_state])


        with tf.variable_scope('q'):        # evaluation network
            self.l_eval = tf.layers.dense(self.tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
            self.q = tf.layers.dense(self.l_eval, self.N_action, kernel_initializer=tf.random_normal_initializer(0,0.1))
        with tf.variable_scope('q_next'):   # target network, not to train
            self.l_target = tf.layers.dense(self.tf_s_, 10, tf.nn.relu, trainable=False)
            self.q_next = tf.layers.dense(self.l_target, self.N_action, trainable=False)

        self.q_target = self.tf_r + config.GAMMA * tf.reduce_max(self.q_next, axis=1)                   # shape=(None, ),

        self.a_indices = tf.stack([tf.range(tf.shape(self.tf_a)[0], dtype=tf.int32), self.tf_a], axis=1)
        self.q_wrt_a = tf.gather_nd(params=self.q, indices=self.a_indices)     # shape=(None, ), q for current state

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_wrt_a))
        self.train_op = tf.train.AdamOptimizer(config.LR).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.Saver = tf.train.Saver()

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < config.EPSILON:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q, feed_dict={self.tf_s: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.N_action)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.MEMORY_COUNTER % config.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.MEMORY_COUNTER += 1

    def learn(self):
        if self.LEARNING_STEP_COUNTER % config.TARGET_REPLACE_ITER == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        self.LEARNING_STEP_COUNTER += 1

        # learning
        sample_index = np.random.choice(config.MEMORY_CAPACITY, config.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :self.N_state]
        b_a = b_memory[:, self.N_state].astype(int)
        b_r = b_memory[:, self.N_state + 1]
        b_s_ = b_memory[:, -self.N_state:]
        self.sess.run(self.train_op, {self.tf_s: b_s, self.tf_a: b_a, self.tf_r: b_r, self.tf_s_: b_s_})

    def save_weight(self, path):
        self.Saver.save(self.sess, path + 'DQN')

    def load_weight(self, path):
        self.saver = tf.train.import_meta_graph(path + 'DQN.meta')
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


# print('\nCollecting experience...')
# for i_episode in range(400):
#     s = env.reset()
#     ep_r = 0
#     while True:
#         env.render()
#         a = choose_action(s)
#
#         # take action
#         s_, r, done, info = env.step(a)
#
#         # modify the reward
#         x, x_dot, theta, theta_dot = s_
#         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         r = r1 + r2
#
#         store_transition(s, a, r, s_)
#
#         ep_r += r
#         if MEMORY_COUNTER > MEMORY_CAPACITY:
#             learn()
#             if done:
#                 print('Ep: ', i_episode,
#                       '| Ep_r: ', round(ep_r, 2))
#
#         if done:
#             break
#         s = s_
#
#
