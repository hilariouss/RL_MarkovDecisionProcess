import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

def discounted_future_reward(reward):
    discount_factor = 0.99

    element_reward = 0
    cumulative_reward = np.zeros_like(reward)
    for k in reversed(range(0, reward.size)):
        element_reward = element_reward * discount_factor + reward[k]
        cumulative_reward[k] = element_reward
    return cumulative_reward

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # placeholder (state)
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        # placeholder (action, reward)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        # Neural network
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None,\
                                      activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, biases_initializer=None,\
                                      activation_fn=tf.nn.softmax)

        # output(action, softmax)
        self.chosen_action = tf.argmax(self.output, 1)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        # tf.range(start, limit) --> start부터 limit까지 number sequence 생성
        # + self.action_holder를 함으로써, 나중에 길게 output을 폈을때 인덱스로 특정 상황에 대한 행동을 추출하기 위함임.
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        # 쫙 펴서 해당 인덱스에 대해서만 유효 output으로.

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []

        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+ '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        update = optimizer.minimize(self.loss, tvars)