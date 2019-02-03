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

        for idx, var in enumerate(tvars): # (index, tvars요소)
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars) # loss(y)들을 각 tvar(x)s로 미분.

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

tf.reset_default_graph()
Agent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8)
total_episode = 5000
max_episodes = 999
update_frequency = 5

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episode:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_episodes):
            a_dist = sess.run(Agent.output, feed_dict={Agent.state_in:[s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1, r, d, _ = env.step(a)
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            if d == True:
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discounted_future_reward(ep_history[:, 2])
                feed_dict = {Agent.reward_holder:ep_history[:, 2],\
                             Agent.action_holder:ep_history[:, 1],\
                             Agent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(Agent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(Agent.gradient_holders,\
                                                      gradBuffer))
                    _ = sess.run(Agent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                total_reward.append(running_reward)
                total_length.append(j)
                break

        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
            i += 1