#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
from argparse import ArgumentParser
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import pprint


# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10


# Configuration file path
DEFAULT_MODEL_SAVEFILE = "./tmp/model"
DEFAULT_CONFIG = "./scenarios/simpler_basic.cfg"

# config_file_path = "../../scenarios/rocket_basic.cfg"
config_file_path = "./scenarios/basic.cfg"

# Converts and down-samples the input image

# Util function
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class Environment:

    def __init__(self, config_file_path):
        self.game = self.initialize_vizdoom(config_file_path)
        # Action = which buttons are pressed
        self.n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=self.n)]

    # Creates and initializes ViZDoom environment.
    def initialize_vizdoom(self, config_file_path):
        print("Initializing doom...")
        game = vzd.DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.init()
        print("Doom initialized.")
        return game

    def getGame(self):
        return self.game

    def getAvailableActions(self):
        return self.actions
    
class Agent:

    def __init__(self, lr, df, fm, af):
        self.env = Environment(DEFAULT_CONFIG)
        self.game = self.env.getGame()
        self.actions = self.env.getAvailableActions()
        self.num_actions = len(self.actions)
        self.epochs = epochs

        self.discount_factor = df
        self.learning_rate = lr
        self.feature_maps = fm
        self.activation_func = af

        self.session = tf.Session()

        self.learn, self.get_q_values, self.get_best_action = self.create_network_tensorflow()
        init = tf.global_variables_initializer() # ver se é o melhor sitio para estar
        self.session.run(init)

        # Create replay memory which will store the transitions
        self.memory = ReplayMemory(capacity=replay_memory_size)

    def create_network_tensorflow(self):
        # Create the input variables
        s1_ = tf.placeholder(
            tf.float32, [None] + list(resolution) + [1], name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(
            tf.float32, [None, self.num_actions], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=self.feature_maps, kernel_size=[6, 6], stride=[3, 3],
                                                activation_fn=self.activation_func,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=self.feature_maps, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=self.activation_func,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=self.activation_func,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.num_actions, activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
        best_a = tf.argmax(q, 1)

        loss = tf.losses.mean_squared_error(q, target_q_)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def function_learn(s1, target_q):
            feed_dict = {s1_: s1, target_q_: target_q}
            l, _ = self.session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state):
            return self.session.run(q, feed_dict={s1_: state})

        def function_get_best_action(state):
            return self.session.run(best_a, feed_dict={s1_: state})

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

        return function_learn, function_get_q_values, function_simple_get_best_action

    def play(self, verbose=1):
        score = 0
        time_start = time()
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))

            train_episodes_finished = 0
            train_scores = []

            if(verbose):
                print("Training...")

            self.game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                self.perform_learning_step(epoch)
                if self.game.is_episode_finished():
                    score = self.game.get_total_reward()
                    train_scores.append(score)
                    self.game.new_episode()
                    train_episodes_finished += 1

            if(verbose):
                print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            if(verbose):
                print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            if(verbose):
                print("\nTesting...")

            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                self.game.new_episode()
                while not self.game.is_episode_finished():
                    state = preprocess(self.game.get_state().screen_buffer)
                    best_action_index = self.get_best_action(state)

                    self.game.make_action(self.actions[best_action_index], frame_repeat)
                r = self.game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            score = test_scores.mean()
            if(verbose):
                print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                "max: %.1f" % test_scores.max())


            if(verbose):
                print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

            #print("Score: " + score + " -> lr: " + str(self.learning_rate) + " | fm: " + str(self.feature_maps) + " | af: " + str(self.activation_func) + " | df:" + str(self.discount_factor))    

        self.game.close()

        if(verbose):
            print("======================================")

        return score    


    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(batch_size)

            q2 = np.max(self.get_q_values(s2), axis=1)
            target_q = self.get_q_values(s1)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + \
                self.discount_factor * (1 - isterminal) * q2
            self.learn(s1, target_q)    

    def perform_learning_step(self, epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(total_epochs, epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * total_epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * total_epochs  # 60% of learning time

            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch - const_eps_epochs) / \
                                (eps_decay_epochs - const_eps_epochs) * \
                    (start_eps - end_eps)
            else:
                return end_eps

        s1 = preprocess(self.game.get_state().screen_buffer)

        # With probability eps make a random action.
        eps = exploration_rate(self.epochs, epoch)
        if random() <= eps:
            a = randint(0, self.num_actions - 1)
        else:
            # Choose the best action according to the network.
            a = self.get_best_action(s1)
        reward =self.game.make_action(self.actions[a], frame_repeat)

        isterminal = self.game.is_episode_finished()
        s2 = preprocess(self.game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        self.learn_from_memory()    


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]



# global variables
# Q-learning settings
learning_rate = 0.0000000001
discount_factor = 0.5
feature_maps = 8
activation_func = tf.nn.softmax
epochs = 5
learning_steps_per_epoch = 1000
replay_memory_size = 10000

if __name__ == "__main__":
    agent = Agent(learning_rate, discount_factor, feature_maps, activation_func)
    score = agent.play(verbose=1)
