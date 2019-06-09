import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from datetime import datetime
from Memory import *


import gym #to be removed
import matplotlib.pyplot as plt


class DDQN():
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10,
                 name='QNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(max_size=1000)
        self.gamma = 0.95    # discount rate
        self.step = 0
        self.lossQ = 0

         # Exploration parameters
        self.explore_start = 1.0  # exploration probability at start
        self.explore_stop = 0.01  # minimum exploration probability
        self.decay_rate = 0.0001  # exponential decay rate for exploration prob

        self.learning_rate = 0.001

        self.config = tf.ConfigProto(intra_op_parallelism_threads=0,
                                     inter_op_parallelism_threads=0,
                                     allow_soft_placement=True,
                                     device_count={'CPU': 1, 'GPU': 1},
                                     log_device_placement=True)



        # now = datetime.utcnow().strftime("%Y%m%d")
        root_logdir = "savedModels"
        self.logdir = "{}/run-{}/".format(root_logdir, "DQN")
        print("The log file is located at:" + self.logdir)
        self.save_path = self.logdir + "DDQN.ckpt"
        self.comile_model_tf(name=name, hidden_size = hidden_size)
        self.session_init()


    def session_init(self):
        self.sess = tf.Session(config=self.config)
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        try:
            self.saver.restore(self.sess, self.save_path)  # or better, use save_path
        except:
            print("Restoring model failed...")

    def comile_model_tf(self, name, hidden_size):
        tf.reset_default_graph()
        # state inputs to the Q-network
        with tf.variable_scope(name):
            with tf.name_scope("input"):
                self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')



            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            with tf.name_scope("DNN"):
            # ReLU hidden layers
                self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
                self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

                # Linear output layer
                self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,
                                                                activation_fn=None)

            with tf.name_scope("Action"):
                # One hot encode the actions to later choose the Q-value for the action
                self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
                one_hot_actions = tf.one_hot(self.actions_, action_size)

            with tf.name_scope("QValue"):
                ### Train with loss (targetQ - Q)^2
                # output has length 2, for two actions. This next line chooses
                # one value from output (per row) according to the one-hot encoded actions.
                self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

                self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
                self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def getLoss(self):
        # self.lossQ = self.sess.run(self.loss)
        return self.lossQ


    # def shuffle_batch(self, X, y, batch_size):
    #     rnd_idx = np.random.permutation(len(X))
    #     n_batches = len(X) // batch_size
    #     for batch_idx in np.array_split(rnd_idx, n_batches):
    #         X_batch, y_batch = X[batch_idx], y[batch_idx]
    #         yield X_batch, y_batch

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        explore_p = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.step)
        self.step += 1
        if explore_p > np.random.rand():
            # Make a random action
            return random.randrange(self.action_size)
        else:
            # Get action from Q-network
            feed = {self.inputs_: state.reshape((1, *state.shape))}
            Qs = self.sess.run(self.output, feed_dict=feed)
            action = np.argmax(Qs)
        return action

    def prediction(self, state):
        # Get action from Q-network
        feed = {self.inputs_: state.reshape((1, *state.shape))}
        Qs = self.sess.run(self.output, feed_dict=feed)
        return np.argmax(Qs)

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        # Train network
        target_Qs = self.sess.run(self.output, feed_dict={self.inputs_: next_states})

        # Set target_Qs to 0 for states where episode ends
        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0)

        targets = rewards + gamma * np.max(target_Qs, axis=1)
        # targets = np
        self.lossQ, _ = self.sess.run([self.loss, mainQN.opt],
                           feed_dict={self.inputs_: states,
                                      self.targetQs_: targets,
                                      self.actions_: actions})

    def save(self):
        self.saver





if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # agent = DDQN(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False

    train_episodes = 1000  # max number of episodes to learn from
    max_steps = 200  # max steps in an episode
    gamma = 0.99  # future reward discount

   

    # Network parameters
    hidden_size = 64  # number of units in each Q-network hidden layer
    learning_rate = 0.0001  # Q-network learning rate

    # Memory parameters
    memory_size = 10000  # memory capacity
    batch_size = 20  # experience mini-batch size
    pretrain_length = batch_size  # number experiences to pretrain the memory

    tf.reset_default_graph()
    mainQN = DDQN(name='DQN', hidden_size=hidden_size, learning_rate=learning_rate)

    # Initialize the simulation
    env.reset()
    # Take one random step to get the pole and cart moving
    state, reward, done, _ = env.step(env.action_space.sample())

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        # env.render()

        # Make a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            mainQN.memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, done, _ = env.step(env.action_space.sample())
        else:
            # Add experience to memory
            mainQN.memory.add((state, action, reward, next_state))
            state = next_state

    rewards_list = []

    step = 0
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        while t < max_steps:
            step += 1

            action = mainQN.act(state)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps

                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(mainQN.getLoss()))
                rewards_list.append((ep, total_reward))

                # Add experience to memory
                mainQN.memory.add((state, action, reward, next_state))

                # Start new episode
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                # Add experience to memory
                mainQN.memory.add((state, action, reward, next_state))
                state = next_state
                t += 1

            # Sample mini-batch from memory
            mainQN.replay(batch_size)


    # saver.save(sess, "checkpoints/cartpole.ckpt")


    test_episodes = 10
    test_max_steps = 400
    env.reset()
    # with tf.Session() as sess:
    #     saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    #
    for ep in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render()
            action = mainQN.prediction(state)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)

            if done:
                t = test_max_steps
                print(t)
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                t += 1

    env.close()
