import numpy as np
import random
import tensorflow as tf
import tf_util as U
import time
from collections import defaultdict

random.seed(30)
tf.set_random_seed(30)

class Learner():
    def __init__(self, env, tf_session, agent_name, obs_shape_n, num_action, agent_id, gamma, epsilon):
        self.env=env
        self.gamma=gamma
        self.obs_shape_n=obs_shape_n[0]
        self.agent_name=agent_name
        self.agent_id=agent_id
        self.num_action=num_action
        self.policy_variables=[]
        self.epsilon=epsilon
        self.session=tf_session
        self.function_name="policy"
        self.alpha = 0.8
        # self.local_policy=self.generate_local_policy()
        self.environment = []
        self.observation=[]
        self.actions=[]
        self.reward=[]
        self.Q = defaultdict(int)
        #self.gamma=0.6
        self._build_net()
        #self.policy_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.agent_name)
        # self.get_policy_variables()
        # self.get_loss()



    def get_local_weights(self):
        return self.session.run(self.policy_variables)

    def set_weights(self,weights):
        for i, weight in enumerate(weights):
            tf.keras.backend.set_value(self.policy_variables[i], weight)

    def reset_episode(self):
        self.observation=[]
        self.actions=[]
        self.environment = []

    def _build_net(self):

        #with tf.name_scope('inputs'):
        self.tf_obs = tf.placeholder("float", shape=[None, self.obs_shape_n], name="observations")
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer1 = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            #name='fc1'+self.agent_name
        )

        # fc2
        layer2 = tf.layers.dense(
            inputs=layer1,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            #name='fc2'+self.agent_name
        )

        # fc3
        all_act = tf.layers.dense(
            inputs=layer2,
            units=self.num_action,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            #name='fc3'+self.agent_name
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

    #with tf.name_scope('loss'):
        # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
        # or in this way:
        # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)  # reward guided loss

    #with tf.name_scope('train'):
        #with tf.variable_scope(self.agent_name):
        self.train_op = tf.train.AdamOptimizer(self.epsilon).minimize(self.loss)


    def action(self,obs):
        prob_weights = self.session.run(self.all_act_prob, feed_dict={self.tf_obs: obs})
        #print(prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        #print(action)
        return action

    def update_Q(self):
        pre_record = self.environment[0]
        n = len(self.environment)
        Q_value = []

        for j in range(1, n):
            record = self.environment[j]
            pre_obs, pre_neighbor_action, pre_reward = pre_record[0], pre_record[1], pre_record[2]
            pre_input = tuple(pre_obs.tolist() + pre_neighbor_action)

            cur_obs, cur_neighbor_action, cur_reward = record[0], record[1], record[2]
            cur_input = tuple(cur_obs.tolist() + cur_neighbor_action)

            self.Q[pre_input] = (1- self.alpha) * self.Q[pre_input] + self.alpha * (pre_reward + self.gamma * self.Q[cur_input])

            pre_record = record

        for k in range(n):
            record = self.environment[k]
            cur_obs, cur_neighbor_action = record[0], record[1]
            cur_input = tuple(cur_obs.tolist() + cur_neighbor_action)
            Q_value.append(self.Q[cur_input])
            self.observation.append(cur_obs)
            self.actions.append(cur_neighbor_action[0])

        return Q_value


    def learn(self, neighbor_Q_value):
        #self.discount_value()
        #print(self.session.run(self.neg_log_prob, feed_dict={self.tf_obs:self.observation, self.tf_acts:self.actions, self.tf_vt:neighbor_Q_value}))
        #print('action len', len(self.actions))
        self.session.run(self.train_op, feed_dict={self.tf_obs:self.observation, self.tf_acts:self.actions, self.tf_vt:neighbor_Q_value})
        #print(self.agent_name,self.session.run(self.loss,feed_dict={self.tf_obs:self.observation,self.tf_acts:self.actions,self.tf_vt:self.discounted_value}))
