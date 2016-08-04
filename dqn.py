__author__ = 'witwolf'

import tensorflow as tf
import numpy as np
import random

from collections import deque


class DQN(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 batch_size=64,
                 gamma=0.9,
                 buffer_size=1024 * 1024,
                 initial_epsilon=0.5,
                 final_epsilon=0.01,
                 logdir='/data/log'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer_size = batch_size

        self.create_Q_network()
        self.create_training_method()

        self.reward = tf.placeholder(tf.float32)
        tf.scalar_summary("reward", self.reward)
        self.merged = tf.merge_all_summaries()

        self.session = tf.InteractiveSession()

        self.summary_writer = tf.train.SummaryWriter(logdir, self.session.graph)
        self.session.run(tf.initialize_all_variables())

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def create_Q_network(self):
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        self.state_input = tf.placeholder("float", [None, self.state_dim])
        hide_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.Q_value = tf.matmul(hide_layer, W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def train_Q_network(self):
        self.time_step += 1
        batch = random.sample(self.replay_buffer, self.batch_size)

        state_batch = [v[0] for v in batch]
        action_batch = [v[1] for v in batch]
        reward_batch = [v[2] for v in batch]
        next_state_batch = [v[3] for v in batch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={
            self.state_input: next_state_batch
        })
        for i in range(0, self.batch_size):
            terminate = batch[i][4]
            if terminate:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def observe_action(self, state, action, reward, next_state, terminate):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, terminate))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > self.batch_size:
            self.train_Q_network()

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def egreedy_action(self, state):

        self.epsilon -= (self.initial_epsilon - self.final_epsilon) / 10000

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return self.action(state)

    def summary(self, step, reward):
        summary = self.session.run(self.merged, feed_dict={self.reward: reward})
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()
