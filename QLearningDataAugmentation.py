import numpy as np
import  sys
from numpy.core._multiarray_umath import ndarray
import funcs as fn
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy.random as rnd
from tensorflow.contrib.layers import fully_connected
import scipy
import plotly.graph_objs as go
import plotly.io as pio
import plotly
import copy
from scipy.io import savemat
from scipy.io import loadmat
from collections import deque
from tqdm import tqdm


class dynamicDNR:

    def __init__(self, testing_week_starts=25, training_week_starts=0):
        self.Feeder_size = 16
        self.num_edge = 13  # number of edges in any of the spanning forest, (reference nodes are not merged)
        self.num_edge_full = 16  # number of edges in the original full meshed system (reference nodes are not merged)
        self.num_node = 16  # total number of buses
        self.num_nonref_node = 13  # number of load buses
        Loss_allstate = np.loadtxt("loss.txt")
        Volt_allstate = np.load("volt.npy")
        Volt_allstate[np.isnan(Volt_allstate)] = 0
        config = fn.obtain_feasible_configuration(self.Feeder_size)
        self.num_outputs = config.shape[0]
        self.num_ref_node = self.num_node - self.num_nonref_node
        P, Q, Sbase, tran_selected = fn.obtain_load_time_series(self.Feeder_size, 1)
        _, _, v2ref = fn.obtain_network_data(self.Feeder_size)
        vref = np.sqrt(v2ref)
        volt_dev = 0.1

        # training data
        P_train = P[:, training_week_starts:testing_week_starts, :]
        Q_train = Q[:, training_week_starts:testing_week_starts, :]
        Loss_allstate_train = Loss_allstate[training_week_starts * 7 * 24:testing_week_starts * 7 * 24, :]
        Volt_allstate_train = Volt_allstate[training_week_starts * 7 * 24:testing_week_starts * 7 * 24, :, :]

        # testing data
        P_test = P[:, testing_week_starts:, :]
        Q_test = Q[:, testing_week_starts:, :]
        Loss_allstate_test = Loss_allstate[testing_week_starts * 7 * 24:, :]
        Volt_allstate_test = Volt_allstate[testing_week_starts * 7 * 24:, :, :]

        # generate the set of historical configuration
        #   a. never produces non-convergent load flow
        #   b. on node 7,12,16 (correspond to python index 6,11,15, respectively) of Fig.4 of
        #      [NETWORK RECONFIGURATION OF DISTRIBUTION SYSTEMS USING DIFFERENTIAL EVOLUTION],
        #      the voltages are within acceptable range.
        config_hist_set_simulation = np.arange(self.num_outputs)
        for ii in range(Loss_allstate.shape[0]):
            config_hist_set_simulation = np.intersect1d(config_hist_set_simulation, np.where(Loss_allstate[ii, :] < 0.9)[0])
            # if the power flow is non-convergent, the loss is 1 (>0.9)
        for ii in range(Volt_allstate.shape[0]):
            config_hist_set_simulation = np.intersect1d(config_hist_set_simulation, np.where(
                np.logical_and(vref - volt_dev <= Volt_allstate[ii, :, 6], Volt_allstate[ii, :, 6] <= vref + volt_dev))[0])
            config_hist_set_simulation = np.intersect1d(config_hist_set_simulation, np.where(
                np.logical_and(vref - volt_dev <= Volt_allstate[ii, :, 11],Volt_allstate[ii, :, 11] <=vref + volt_dev))[0])
            config_hist_set_simulation = np.intersect1d(config_hist_set_simulation, np.where(
                np.logical_and(vref - volt_dev <= Volt_allstate[ii, :, 15],Volt_allstate[ii, :, 15] <=vref + volt_dev))[0])
        # generate historical configuration
        # (1). generate the configuration historical data. We assume it is a Markov process with
        # transition probability config_prob_tran
        # prob: Markov transition model diagonal ones
        prob = 0.9
        config_hist = np.zeros(Loss_allstate.shape[0], dtype=int)
        config_hist[0] = np.random.choice(config_hist_set_simulation)  # random initial
        config_prob_tran = np.identity(config_hist_set_simulation.size) * prob + np.ones(
            (config_hist_set_simulation.size, config_hist_set_simulation.size)) * (1 - prob) / config_hist_set_simulation.size
        for ii in range(1, Loss_allstate.shape[0]):
            config_hist[ii] = np.random.choice(config_hist_set_simulation,
                                               p=config_prob_tran[np.where(config_hist[ii - 1] == config_hist_set_simulation),:].squeeze())
        # (2). historical configuration generation complete
        config_hist_test = config_hist[testing_week_starts * 7 * 24:]
        config_hist_train = config_hist[training_week_starts * 7 * 24:testing_week_starts * 7 * 24]
        config_hist_train = config_hist_train.reshape((-1, 168)).T  # reshape into [t, episode] format

        config_hist_set = np.unique(config_hist_train)

        # config_nn_encoding is a matrix of shape num_radial_topology * num_edges. Each row is a configuration
        # for example, config_nn_encoding[ii,:] = [on-off-status of each edge]
        # on_value and off_value need to better match neural network initial weights
        on_value = 0.1
        off_value = 0.0
        config_nn_encoding = np.ones(shape=(config.shape[0], self.num_edge_full)) * off_value
        for ii in range(config.shape[0]):
           for jj in range(config.shape[1]):
               config_nn_encoding[ii, config[ii, jj]] = on_value

        # prepare the training and testing data for both gaussian process and neural network
        self.config_sel = config_hist_set

        Cost_per_kWh = 0.13  # average retail electricity price (in dollar/kWh)
        Cost_per_switch = 4.6  # dollar per switching action
        Cost_per_pu = Cost_per_kWh * Sbase  # in dollar/pu
        w_switch = Cost_per_switch / Cost_per_pu  # excessive switching action penalty coefficient
        lambda_voltage_violation = 50  # the (fixed) Lagrange multiplier for voltage and current violation (in the CMDP)
                                       # the numerical value of lambda_voltage_violation is 50 since we scale the reward
                                       # by 50 in our Q learning neural network. We scale this lambda by the same factor
                                       # to indicate that the lambda is the unity (1) if we had not scale the reward by 50
        # training data
        self.ObsIdxPt_gp = np.zeros((0, self.num_nonref_node * 2 + self.num_edge_full))
        self.ObsIdxPt_rl = np.zeros((0, 1 + self.num_nonref_node * 2 + self.num_edge_full))
        self.ObsIdxPtp1_rl = np.zeros((0, 1 + self.num_nonref_node * 2 + self.num_edge_full))
        self.FeatureTrain_nn = np.zeros((0, self.num_nonref_node * 2))
        self.ConfigTrain_nn = np.zeros((0), dtype=np.int32)
        self.ConfigTrain_rl = np.zeros((0), dtype=np.int32)
        self.Obs = np.array([])
        self.ObsRL = np.array([])
        self.ContinueTrain = np.array([])

        # testing data
        self.IdxPt_gp = np.zeros((0, self.num_nonref_node * 2 + self.num_edge_full))
        self.IdxPt_rl = np.zeros((0, 1 + self.num_nonref_node * 2 + self.num_edge_full))
        self.IdxPtp1_rl = np.zeros((0, 1 + self.num_nonref_node * 2 + self.num_edge_full))
        self.FeatureTest_nn = np.zeros((0, self.num_nonref_node * 2))
        self.ConfigTest_nn = np.zeros((0), dtype=np.int32)
        self.ConfigTest_rl = np.zeros((0), dtype=np.int32)
        self.ObsTest = np.array([])
        self.ObsTestRL = np.array([])
        self.ContinueTest = np.array([])
        # generating the training and testing historical data sets. We assume the testing data are immediately after the
        # training data
        state_ID = 0  # the label 0,1,2,3... for each sample of the historical data set
        action_prev = self.config_sel[1]  # a random starting action
        T = P_train.shape[0]
        for episode in range(0, P_train.shape[1]):  # for all training episodes
            for t in range(0, P_train.shape[0]):
                action = config_hist_train[t, episode]
                r = -Loss_allstate[state_ID, action] * 50
                tempnn = np.hstack((P_train[t, episode, :], Q_train[t, episode, :]))
                tempgp = np.hstack((P_train[t, episode, :], Q_train[t, episode, :], config_nn_encoding[action, :]))
                temprl = np.hstack((t/T, P_train[t, episode, :], Q_train[t, episode, :], config_nn_encoding[action_prev, :]))
                if t == P_train.shape[0]-1:
                    temprlp1 = temprl
                else:
                    temprlp1 = np.hstack(((t+1)/T, P_train[t+1, episode, :], Q_train[t+1, episode, :], config_nn_encoding[action, :]))
                self.ObsIdxPt_gp = np.vstack((self.ObsIdxPt_gp, tempgp))
                self.ObsIdxPt_rl = np.vstack((self.ObsIdxPt_rl, temprl))
                self.ObsIdxPtp1_rl = np.vstack((self.ObsIdxPtp1_rl, temprlp1))
                self.FeatureTrain_nn = np.vstack((self.FeatureTrain_nn, tempnn))
                self.ConfigTrain_nn = np.append(self.ConfigTrain_nn, action)
                self.ConfigTrain_rl = np.append(self.ConfigTrain_rl, action)
                self.Obs = np.append(self.Obs, r)
                self.ObsRL = np.append(self.ObsRL, r - 50*(w_switch * np.setdiff1d(config[action, :], config[action_prev, :]).size * 2)
                                      - lambda_voltage_violation * np.sum(
                    (np.absolute(Volt_allstate[state_ID, action, [6, 11, 15]] - vref) - volt_dev).clip(min=0)))
                if t+1 == P_train.shape[0]-1:  # if t+1 is the episode terminate time:
                    self.ContinueTrain = np.append(self.ContinueTrain, 0)
                else:
                    self.ContinueTrain = np.append(self.ContinueTrain, 1)
                state_ID += 1
                action_prev = action
        for episode in range(0, P_test.shape[1]):
            for t in range(0, P_test.shape[0]):
                action = config_hist_test[t]
                r = -Loss_allstate[state_ID, action] * 50
                tempnn = np.hstack((P_test[t, episode, :], Q_test[t, episode, :]))
                tempgp = np.hstack((P_test[t, episode, :], Q_test[t, episode, :], config_nn_encoding[action, :]))
                temprl = np.hstack((t/T, P_test[t, episode, :], Q_test[t, episode, :], config_nn_encoding[action_prev, :]))
                if t == P_test.shape[0]-1:
                    temprlp1 = temprl
                else:
                    temprlp1 = np.hstack(((t+1)/T, P_test[t+1, episode, :], Q_test[t+1, episode, :], config_nn_encoding[action, :]))
                self.IdxPt_gp = np.vstack((self.IdxPt_gp, tempgp))
                self.IdxPt_rl = np.vstack((self.IdxPt_rl, temprl))
                self.IdxPtp1_rl = np.vstack((self.IdxPtp1_rl, temprlp1))
                self.FeatureTest_nn = np.vstack((self.FeatureTest_nn, tempnn))
                self.ConfigTest_nn = np.append(self.ConfigTest_nn, action)
                self.ConfigTest_rl = np.append(self.ConfigTest_rl, action)
                self.ObsTest = np.append(self.ObsTest, r)
                self.ObsTestRL = np.append(self.ObsTestRL, r - 50 * (w_switch * np.setdiff1d(config[action, :], config[action_prev, :]).size * 2)
                                          - lambda_voltage_violation * np.sum(
                    (np.absolute(Volt_allstate[state_ID, action, [6, 11, 15]] - vref) - volt_dev).clip(min=0)))
                if t+1 == P_test.shape[0]-1:  # if t+1 is the episode terminate time:
                    self.ContinueTest = np.append(self.ContinueTest, 0)
                else:
                    self.ContinueTest = np.append(self.ContinueTest, 1)
                state_ID += 1
                action_prev = action
        # generate some augmented data
        prob_aug = 0.8
        self.ObsIdxPtAug_gp = np.zeros((0, self.num_nonref_node * 2 + self.num_edge_full))
        self.ObsIdxPtAug_rl = np.zeros((0, 1 + self.num_nonref_node * 2 + self.num_edge_full))
        self.ObsIdxPtp1Aug_rl = np.zeros((0, 1 + self.num_nonref_node * 2 + self.num_edge_full))
        self.FeatureAug_nn = np.zeros((0, self.num_nonref_node * 2))
        self.ConfigAug_nn = np.zeros((0), dtype=np.int32)
        self.ConfigAug_rl = np.zeros((0), dtype=np.int32)
        self.ObsAug = np.array([])  # ObsAug is not allowed to perform train. It is for error assessment
        self.ObsAugRL = np.array([])
        self.ContinueAug = np.array([])

        # synthetic historical config
        # (1). generate the configuration historical data. We assume it is a Markov process with
        # transition probability config_prob_tran
        # prob: Markov transition model diagonal ones
        config_hist_aug = np.zeros(Loss_allstate.shape[0], dtype=int)
        config_hist_aug[0] = np.random.choice(config_hist_set)  # random initial
        config_prob_tran = np.identity(config_hist_set.size) * prob_aug + np.ones(
            (config_hist_set.size, config_hist_set.size)) * (1 - prob_aug) / config_hist_set.size
        for ii in range(1, Loss_allstate.shape[0]):
            config_hist_aug[ii] = np.random.choice(config_hist_set,
                                               p=config_prob_tran[np.where(config_hist_aug[ii-1] == config_hist_set),:].squeeze())
        # (2). historical configuration generation complete
        config_hist_aug = config_hist_aug.reshape((-1, 168)).T  # reshape into [t, episode] format

        # (3). put into the augmented dataset
        state_ID = 0  # the label 0,1,2,3... for each sample of the historical data set
        action_prev = self.config_sel[1]
        for episode in range(0, P_train.shape[1]):  # for all training episodes
            for t in range(0, P_train.shape[0]):
                action = config_hist_aug[t, episode]
                if np.any(np.isin(self.config_sel, action)):
                    r = -Loss_allstate[state_ID, action] * 50
                    tempnn = np.hstack((P_train[t, episode, :], Q_train[t, episode, :]))
                    tempgp = np.hstack((P_train[t, episode, :], Q_train[t, episode, :], config_nn_encoding[action, :]))
                    temprl = np.hstack((t/T, P_train[t, episode, :], Q_train[t, episode, :], config_nn_encoding[action_prev, :]))
                    if t == P_train.shape[0]-1:
                        temprlp1 = temprl
                    else:
                        temprlp1 = np.hstack(((t+1)/T, P_train[t+1, episode, :], Q_train[t+1, episode, :], config_nn_encoding[action, :]))
                    self.ObsIdxPtAug_gp = np.vstack((self.ObsIdxPtAug_gp, tempgp))
                    self.ObsIdxPtAug_rl = np.vstack((self.ObsIdxPtAug_rl, temprl))
                    self.ObsIdxPtp1Aug_rl = np.vstack((self.ObsIdxPtp1Aug_rl, temprlp1))
                    self.FeatureAug_nn = np.vstack((self.FeatureAug_nn, tempnn))
                    self.ConfigAug_nn = np.append(self.ConfigAug_nn, action)
                    self.ConfigAug_rl = np.append(self.ConfigAug_rl, action)
                    self.ObsAug = np.append(self.ObsAug, r)  # ObsAug is not allowed to perform train. It is for error assessment
                    self.ObsAugRL = np.append(self.ObsAugRL, - 50 * (w_switch * np.setdiff1d(config[action, :], config[action_prev, :]).size * 2))
                    if t+1 == P_train.shape[0]-1:  # if t+1 is the episode terminate time:
                        self.ContinueAug = np.append(self.ContinueAug, 0)
                    else:
                        self.ContinueAug = np.append(self.ContinueAug, 1)
                    # we delibrately set the ObsAugRL to be without r here because it has to be filled by the gaussian
                    # process estimator
                state_ID += 1
                action_prev = action

        # discount factor
        self.discount_factor = 0.95

        # the followings are necessary for the Q learning agent
        self.config_nn_encoding = config_nn_encoding
        self.config_hist_set = config_hist_set
        self.config_hist_train = config_hist_train
        self.Loss_allstate_test = Loss_allstate_test
        self.Loss_allstate_train = Loss_allstate_train
        self.config = config
        self.Cost_per_pu = Cost_per_pu
        self.Cost_per_switch = Cost_per_switch
        self.Volt_allstate_test = Volt_allstate_test

        # calculate the original cost of the historical data
        state_ID_test = 0
        total_switching_num_week_ori = 0
        total_loss_week_ori = 0
        worst_volt_week_ori = 1.0
        action_prev = config_hist_train[-1, -1]
        for tt in range(0, T):

            action = config_hist_test[tt]
            total_switching_num_week_ori += np.setdiff1d(config[action_prev, :], config[action, :]).size * 2
            action_prev = action

            loss = Loss_allstate_test[state_ID_test, action]
            total_loss_week_ori += loss

            volt = np.min(Volt_allstate_test[state_ID_test, action, [6,11,15]])
            worst_volt_week_ori = np.minimum(worst_volt_week_ori, volt)
            state_ID_test += 1

        self.total_money_ori = total_loss_week_ori * Cost_per_pu + total_switching_num_week_ori * Cost_per_switch
        self.minimum_volt_ori = worst_volt_week_ori

        # calculate the optimal cost of the historical injection patterns (solved using dynamic programming)
        # for implementation convenient, we pre-solve the switching cost table
        D_switching_cost = np.zeros((config.shape[0], config.shape[0]))
        for ii in range(config.shape[0]):
            for jj in range(config.shape[0]):
                D_switching_cost[ii, jj] = np.setdiff1d(config[ii, :], config[jj, :]).size * 2
        D_switching_cost = D_switching_cost * Cost_per_switch
        total_switching_num_week_opt = 0
        total_loss_week_opt = 0
        ValueTable = np.zeros((168 + 1, 190))  # state-value table
        ConsVioTable = np.zeros((168 + 1, 190), dtype=np.int32)  # voltage limit constraint violation index. 1-the constraint is violated, 0-the constraint is not violated
        for state_ID_test in range(167 + 1, -1, -1):  # backward trace
            if state_ID_test == 168:
                for jj in range(190):
                    ValueTable[state_ID_test, jj] = 0
            else:
                for jj in range(190):
                    # update the constraint violation table if there is a constraint violation
                    if np.sum((np.absolute(Volt_allstate_test[state_ID_test, jj, [6, 11, 15]] - vref) - volt_dev).clip(min=0)) > 0:
                        ConsVioTable[state_ID_test, jj] = 1  # 1: the constraint is violated
                # update the action value table
                loss_all_cost = Loss_allstate_test[state_ID_test, :] * Cost_per_pu  # convert to dollars
                reward_to_go = -D_switching_cost - loss_all_cost.reshape((loss_all_cost.size, -1)) + ValueTable[state_ID_test + 1, :]
                for jj in range(190):
                    ValueTable[state_ID_test, jj] = np.max(
                        reward_to_go[jj, np.where(ConsVioTable[state_ID_test + 1, :] == 0)[0]])
        reward_to_go = -D_switching_cost[config_hist_train[-1, -1], :] + ValueTable[0, :]
        total_money_opt = np.max(reward_to_go[np.where(ConsVioTable[0, :] == 0)[0]])
        self.total_money_opt = -total_money_opt
        # find the optimal action sequence to determine the worst voltage
        worst_volt_week_opt = 1.0
        for state_ID_test in range(168):
            action = np.argmax(ValueTable[state_ID_test, np.where(ConsVioTable[state_ID_test, :] == 0)[0]])  # TODO: this step might be problematic if state_ID_test == 0
            action = np.where(ConsVioTable[state_ID_test, :] == 0)[0][action]
            volt = np.min(Volt_allstate_test[state_ID_test, action, [6,11,15]])
            worst_volt_week_opt = np.minimum(worst_volt_week_opt, volt)
        self.minimum_volt_opt = worst_volt_week_opt

    def AssemblyQLearningData(self, gp, augmentdata='No augmented data'):
        if augmentdata == 'GP augmented data':
            self.S = np.vstack((self.ObsIdxPt_rl, self.ObsIdxPtAug_rl[gp.IdxKeep, :]))
            self.A = np.concatenate((self.ConfigTrain_rl, self.ConfigAug_rl[gp.IdxKeep]))
            self.R = np.concatenate((self.ObsRL, self.ObsAugRL[gp.IdxKeep] + np.squeeze(gp.TargetAugAvg_gp)[gp.IdxKeep]))
            self.Sp = np.vstack((self.ObsIdxPtp1_rl, self.ObsIdxPtp1Aug_rl[gp.IdxKeep, :]))
            self.C = np.concatenate((self.ContinueTrain, self.ContinueAug[gp.IdxKeep]))
        elif augmentdata == 'No augmented data':
            self.S = self.ObsIdxPt_rl
            self.A = self.ConfigTrain_rl
            self.R = self.ObsRL
            self.Sp = self.ObsIdxPtp1_rl
            self.C = self.ContinueTrain
        elif augmentdata == 'True augmented data':
            # self.S = np.vstack((self.ObsIdxPt_rl, self.ObsIdxPtAug_rl[gp.IdxKeep, :]))
            # self.A = np.concatenate((self.ConfigTrain_rl, self.ConfigAug_rl[gp.IdxKeep]))
            # self.R = np.concatenate((self.ObsRL, self.ObsAugRL[gp.IdxKeep] + self.ObsAug[gp.IdxKeep]))
            # self.Sp = np.vstack((self.ObsIdxPtp1_rl, self.ObsIdxPtp1Aug_rl[gp.IdxKeep, :]))
            # self.C = np.concatenate((self.ContinueTrain, self.ContinueAug[gp.IdxKeep]))
            self.S = np.vstack((self.ObsIdxPt_rl, self.ObsIdxPtAug_rl))
            self.A = np.concatenate((self.ConfigTrain_rl, self.ConfigAug_rl))
            self.R = np.concatenate((self.ObsRL, self.ObsAugRL + self.ObsAug))
            self.Sp = np.vstack((self.ObsIdxPtp1_rl, self.ObsIdxPtp1Aug_rl))
            self.C = np.concatenate((self.ContinueTrain, self.ContinueAug))
        else:
            print("expected str inputs: GP augmented data/No augmented data/True augmented data")
        self.PQtest = self.FeatureTest_nn
        self.PQtrain = self.FeatureTrain_nn
        self.Rtest = self.ObsTestRL



class QAgent:
    def __init__(self, problem, agentID, seed=1):
        def q_network(X_state, scope):  # X_state: agent's state
            with tf.variable_scope(scope) as scope:
                # the size of dimension with -1 is computed by tensorflow so that the input and output tensors have
                # matching dimensions
                #hidden_1 = fully_connected(X_state[:, 1:],
                #                         400,
                #                         activation_fn = tf.nn.relu,
                #                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                #                           weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
                #                           biases_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                #                           biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
                hidden_2 = fully_connected(X_state[:, 1:],
                                           #problem.num_outputs*2,
                                            600,
                                           activation_fn=tf.nn.relu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
                                           biases_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                           biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
                outputs = fully_connected(tf.concat([X_state[:, 0:1], hidden_2], axis=1),
                                          problem.num_outputs,
                                          activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
                                          biases_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
            return outputs, trainable_vars_by_name

        self.State_placeholder = tf.placeholder(tf.float32, shape=[None, problem.num_nonref_node*2 + 1 + problem.num_edge_full])
        self.Config_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.Qhat_placeholder = tf.placeholder(tf.float32, shape=[None])

        self.Q_values, self.Q_vars = q_network(self.State_placeholder, scope="q_networks/value"+str(agentID))
        self.Q_targets, self.Q_tars = q_network(self.State_placeholder, scope="q_networks/target"+str(agentID))
        self.copy_ops = [target_var.assign(self.Q_vars[var_name]) for var_name, target_var in self.Q_tars.items()]
        self.copy_value_to_target = tf.group(*self.copy_ops)

        self.Q_values_rl = tf.reduce_sum(self.Q_values * tf.one_hot(self.Config_placeholder, problem.num_outputs), axis=1, keep_dims=False)
        self.cost = tf.reduce_mean(tf.square(self.Qhat_placeholder - self.Q_values_rl))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

    def Train(self, problem, num_steps=12000, batch_size=512, test_steps=3000, copy_steps=30):
        self.dollar_test = np.array([])
        self.dollar_train = np.array([])
        self.min_volt_test = np.array([])
        display_steps = 5000
        for step in tqdm(range(num_steps)):
            if step % test_steps == 0:
                state_ID_test = 0
                total_switching_num_week = 0
                total_loss_week = 0
                worst_volt_week = 1.0
                action_prev = problem.config_hist_train[-1, -1]  # the last configuration before testing week
                T = 168
                for tt in range(0, T):
                    # 1. states
                    state = np.hstack((tt / T, problem.PQtest[tt, :], problem.config_nn_encoding[action_prev, :]))
                    # 2. q values
                    q_values = self.Q_values.eval(feed_dict={self.State_placeholder: np.expand_dims(state, axis=0)})
                    # 3. select action (only select actions in historical dataset)
                    action = np.argmax(q_values[0, problem.config_hist_set])
                    action = problem.config_hist_set[action]
                    # 4. calculate the switching cost:
                    total_switching_num_week += np.setdiff1d(problem.config[action_prev, :], problem.config[action, :]).size * 2
                    action_prev = action
                    # 5. calculate the total network loss:
                    loss = problem.Loss_allstate_test[state_ID_test, action]
                    total_loss_week += loss
                    # 6. calculate the worst voltage:
                    volt = np.min(problem.Volt_allstate_test[state_ID_test, action, [6, 11, 15]])
                    worst_volt_week = np.minimum(worst_volt_week, volt)
                    # 7. update state ID
                    state_ID_test += 1
                total_money_DQN = total_loss_week * problem.Cost_per_pu + total_switching_num_week * problem.Cost_per_switch
                self.dollar_test = np.append(self.dollar_test, total_money_DQN)
                self.min_volt_test = np.append(self.min_volt_test, worst_volt_week)

                # test Q agent performance on one of the training weeks. Choose the last week as the training data
                state_ID_train = 24*168
                total_switching_num_week = 0
                total_loss_week = 0
                action_prev = problem.config_hist_train[-1, -2]
                for tt in range(0, T):
                    # 1. states
                    state = np.hstack((tt / T, problem.PQtrain[state_ID_train, :], problem.config_nn_encoding[action_prev, :]))
                    # 2. q values
                    q_values = self.Q_values.eval(feed_dict={self.State_placeholder: np.expand_dims(state, axis=0)})
                    # 3. select action (only select actions in historical dataset)
                    action = np.argmax(q_values[0, problem.config_hist_set])
                    action = problem.config_hist_set[action]
                    # 4. calculate the switching cost:
                    total_switching_num_week += np.setdiff1d(problem.config[action_prev, :], problem.config[action, :]).size * 2
                    action_prev = action
                    # 5. calculate the total network loss:
                    loss = problem.Loss_allstate_train[state_ID_train, action]
                    total_loss_week += loss
                    # 6. update state ID
                    state_ID_train += 1
                total_money_DQN = total_loss_week * problem.Cost_per_pu + total_switching_num_week * problem.Cost_per_switch
                self.dollar_train = np.append(self.dollar_train, total_money_DQN)

            indices = rnd.permutation(problem.S.shape[0])[:batch_size]
            next_q_values = self.Q_targets.eval(feed_dict={self.State_placeholder: problem.Sp[indices]})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=False)
            y_val = problem.R[indices] + problem.C[indices] * problem.discount_factor * max_next_q_values

            self.train_op.run(feed_dict={self.State_placeholder: problem.S[indices, :],
                                        self.Config_placeholder: problem.A[indices],
                                        self.Qhat_placeholder: y_val})
            if step % copy_steps == 0:
                self.copy_value_to_target.run()

class GaussianProcess:
    # in the Gaussian process setting, a training data is called an "observation" because the Gaussian process is essentially
    # a kernel machine. Therefore "observation" connotes "local activation". Also the feature is called an "index" because
    # a Gaussian process is a generalization of multivariate normal distribution, which we know what index means
    def __init__(self, problem):
        # init method construct a Tensorflow computation graph
        # input: problem: an object of class dynamicDNR
        ObsIdxShape = [None, problem.num_nonref_node * 2 + problem.num_edge_full]
        self.ObsIdxPt_placeholder = tf.placeholder(tf.float64, shape=ObsIdxShape)
        self.Obs_placeholder = tf.placeholder(tf.float64, shape=[None])
        self.IdxPt_placeholder = tf.placeholder(tf.float64, shape=ObsIdxShape)
        #self.Amp = tf.exp(tf.get_variable('amplitude', [1], dtype=np.float64, initializer=tf.constant_initializer(1)))
        #self.Len = tf.exp(tf.get_variable('length_scale', [1], dtype=np.float64, initializer=tf.constant_initializer(1)))
        #self.Var = tf.exp(tf.get_variable('observation_noise_variance', [1], dtype=np.float64, initializer=tf.constant_initializer(-5)))
        self.Amp = tf.get_variable('amplitude', [1], dtype=np.float64, initializer=tf.constant_initializer(0.37), constraint=lambda t: tf.nn.relu(t))
        self.Len = tf.get_variable('length_scale', [1], dtype=np.float64, initializer=tf.constant_initializer(0.12), constraint=lambda t: tf.nn.relu(t))
        self.Var = tf.get_variable('observation_noise_variance', [1], dtype=np.float64, initializer=tf.constant_initializer(1e-40), constraint=lambda t: tf.nn.relu(t))
        self.kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
            amplitude=self.Amp,
            length_scale=self.Len,
            feature_ndims=1)
        self.GPRM_train = tfp.distributions.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=self.ObsIdxPt_placeholder,
            observation_index_points=self.ObsIdxPt_placeholder,
            observations=self.Obs_placeholder,
            observation_noise_variance=self.Var,
            predictive_noise_variance=self.Var,
            jitter=1e-06)
        self.NegLogLikelihood = -self.GPRM_train.log_prob(self.Obs_placeholder)
        self.GPRM_train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.NegLogLikelihood)

        self.GPRM_test = tfp.distributions.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=self.IdxPt_placeholder,
            observation_index_points=self.ObsIdxPt_placeholder,
            observations=self.Obs_placeholder,
            observation_noise_variance=self.Var,
            predictive_noise_variance=self.Var,
            jitter=1e-06)

        self.Sig_train = tf.sqrt(tf.linalg.tensor_diag_part(tf.squeeze(self.GPRM_train.covariance())))
        self.Avg_train = self.GPRM_train.mean()
        self.Sig_test = tf.sqrt(tf.linalg.tensor_diag_part(tf.squeeze(self.GPRM_test.covariance())))
        self.Avg_test = self.GPRM_test.mean()

    def Train(self, problem, num_steps=6000, batch_size=512, test_steps=3000):
        # train the gaussian process for reward prediction and uncertainty estimation
        # input: steps: number of gradient descent steps
        self.LogLikelihood_all = np.array([])
        self.Amp_all = np.array([])
        self.Len_all = np.array([])
        self.Var_all = np.array([])
        for step in tqdm(range(num_steps)):
            if step % test_steps == 0:  # record the training performance
                self.LogLikelihood_all = np.append(self.LogLikelihood_all, -self.NegLogLikelihood.eval(feed_dict={
                                                                        self.ObsIdxPt_placeholder: problem.IdxPt_gp,
                                                                        self.Obs_placeholder: problem.ObsTest}))
                self.Amp_all = np.append(self.Amp_all, self.Amp.eval())
                self.Len_all = np.append(self.Len_all, self.Len.eval())
                self.Var_all = np.append(self.Var_all, self.Var.eval())
            indices = rnd.permutation(problem.ObsIdxPt_gp.shape[0])[:batch_size]
            self.GPRM_train_op.run(feed_dict={self.ObsIdxPt_placeholder: problem.ObsIdxPt_gp[indices, :],
                                              self.Obs_placeholder: problem.Obs[indices]})
    def Test(self, problem):
        self.TargetTestAvg_gp = self.Avg_test.eval(feed_dict={self.IdxPt_placeholder: problem.IdxPt_gp,
                                                              self.ObsIdxPt_placeholder: problem.ObsIdxPt_gp,
                                                              self.Obs_placeholder: problem.Obs})
        self.TargetTestStd_gp = self.Sig_test.eval(feed_dict={self.IdxPt_placeholder: problem.IdxPt_gp,
                                                              self.ObsIdxPt_placeholder: problem.ObsIdxPt_gp,
                                                              self.Obs_placeholder: problem.Obs})
        print("calculating g ... constructing covariance")
        all_points = np.vstack((problem.ObsIdxPt_gp, problem.IdxPt_gp))
        amp_me = self.Amp.eval()
        len_me = self.Len.eval()
        Var_me = self.Var.eval()

        D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(all_points, 'sqeuclidean'))
        cov_me = amp_me ** 2 * np.exp(-D/(2 * len_me ** 2))

        num_obs = problem.ObsIdxPt_gp.shape[0]
        inv_cov = np.linalg.inv(cov_me[:num_obs, :num_obs] + (Var_me+1e-8) * np.eye(num_obs))
        self.TargetTestStd_Mod_K = np.sum(inv_cov)
        print("calculating g ... finding w's")
        w = np.matmul(inv_cov, cov_me[:num_obs,num_obs:])
        self.TargetTestStd_Mod_g = 1 - np.sum(w, axis=0)
        print("finished")
    def TestAugData(self, problem):
        self.TargetAugAvg_gp = self.Avg_test.eval(feed_dict={self.IdxPt_placeholder: problem.ObsIdxPtAug_gp,
                                                              self.ObsIdxPt_placeholder: problem.ObsIdxPt_gp,
                                                              self.Obs_placeholder: problem.Obs})
        self.TargetAugStd_gp = self.Sig_test.eval(feed_dict={self.IdxPt_placeholder: problem.ObsIdxPtAug_gp,
                                                              self.ObsIdxPt_placeholder: problem.ObsIdxPt_gp,
                                                              self.Obs_placeholder: problem.Obs})

        print("calculating g ... constructing covariance")
        all_points = np.vstack((problem.ObsIdxPt_gp, problem.ObsIdxPtAug_gp))
        amp_me = self.Amp.eval()
        len_me = self.Len.eval()
        Var_me = self.Var.eval()

        D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(all_points, 'sqeuclidean'))
        cov_me = amp_me ** 2 * np.exp(-D/(2 * len_me ** 2))

        num_obs = problem.ObsIdxPt_gp.shape[0]
        inv_cov = np.linalg.inv(cov_me[:num_obs, :num_obs] + (Var_me+1e-8) * np.eye(num_obs))
        self.TargetAugStd_Mod_K = np.sum(inv_cov)
        print("calculating g ... finding w's")
        w = np.matmul(inv_cov, cov_me[:num_obs,num_obs:])
        self.TargetAugStd_Mod_g = 1 - np.sum(w, axis=0)
        print("finished")

    def Postprocessing(self):
        gp_aug_std = (np.squeeze(self.TargetAugStd_gp) + np.sqrt(self.TargetAugStd_Mod_K) * np.abs(np.squeeze(self.TargetAugStd_Mod_g))) / 50
        std = gp_aug_std - np.average(gp_aug_std)  # centering
        self.IdxRemove = np.where(std > 3.0*np.std(std))[0]
        self.IdxKeep = np.setdiff1d(np.arange(std.size), self.IdxRemove)


class Experiment1:
    # Experimenet 1: test the performance of the GP and MC dropout algorithms. We generate 1 historical data and 1
    # augmented data, and run the algorithm 10 times and average the results. We report in this experiment
    # 1. the true test and augmented data (network loss)
    # 2. prediction of the test and augmented data (network loss)
    # 3. standard deviation of the test and augmented data (network loss)
    def __init__(self, test_steps=100, method='DQN'):
        self.dollar_Qagent1 = np.array([])
        self.dollar_Qagent2 = np.array([])
        #self.dollar_Qagent3 = np.array([])
        self.dollar_train_Qagent1 = np.array([])
        self.dollar_train_Qagent2 = np.array([])
        #self.dollar_train_Qagent3 = np.array([])
        self.min_volt_Qagent1 = np.array([])
        self.min_volt_Qagent2 = np.array([])
        #self.min_volt_Qagent3 = np.array([])
        self.fig_cnt = 0
        self.test_steps = test_steps
        self.method = method
    def run(self, num_repeats=10, np_seed=1, tf_seed=2):

        # random seeds
        np.random.seed(np_seed)
        tf.reset_default_graph()  # this line of code should go BEFORE the tf.random.set_random_seed(x)
        tf.random.set_random_seed(tf_seed)  # tensorflow graph level seed, see documentation

        # initialize problem
        prob_DNR = dynamicDNR()

        # Gaussian process train
        gp = GaussianProcess(prob_DNR)
        if self.method == 'DQN':
            pass
        else:
            init = tf.global_variables_initializer()  # this line of code should go AFTER the tf graph construction phase
            with tf.Session() as sess:
                init.run()
                gp.Train(prob_DNR, num_steps=10000, batch_size=256, test_steps=10000)
                gp.Test(prob_DNR)
                gp.TestAugData(prob_DNR)
                gp.Postprocessing()

        # train Q agents
        for repeat in range(num_repeats):
            # the three agents have the same neural network initial weights
            Qagent1 = QAgent(prob_DNR, agentID=1+3*repeat, seed=repeat)  # Q agent 1: without augmented data
            Qagent2 = QAgent(prob_DNR, agentID=2+3*repeat, seed=repeat)  # Q agent 2: with augmented data
            #Qagent3 = QAgent(prob_DNR, agentID=3+3*repeat, seed=repeat)  # Q agent 3: with true (unbiased) augmented data
            init = tf.global_variables_initializer()
            print("Run "+str(repeat)+" of "+str(num_repeats))
            with tf.Session() as sess:
                sess.run(init)
                if self.method == 'DQN':
                # train Q agent without augmented data
                    prob_DNR.AssemblyQLearningData(gp, augmentdata='No augmented data')
                    Qagent1.Train(prob_DNR, num_steps=12000, batch_size=64, test_steps=self.test_steps, copy_steps=30)
                    self.dollar_Qagent1 = Qagent1.dollar_test if self.dollar_Qagent1.size == 0 else np.vstack((self.dollar_Qagent1, Qagent1.dollar_test))
                    self.dollar_train_Qagent1 = Qagent1.dollar_train if self.dollar_train_Qagent1.size == 0 else np.vstack((self.dollar_train_Qagent1, Qagent1.dollar_train))
                    self.min_volt_Qagent1 = Qagent1.min_volt_test if self.min_volt_Qagent1.size == 0 else np.vstack((self.min_volt_Qagent1, Qagent1.min_volt_test))
                else:
                    # train Q agent with augmented data
                    prob_DNR.AssemblyQLearningData(gp, augmentdata='GP augmented data')
                    Qagent2.Train(prob_DNR, num_steps=12000, batch_size=64, test_steps=self.test_steps, copy_steps=30)
                    self.dollar_Qagent2 = Qagent2.dollar_test if self.dollar_Qagent2.size == 0 else np.vstack((self.dollar_Qagent2, Qagent2.dollar_test))
                    self.dollar_train_Qagent2 = Qagent2.dollar_train if self.dollar_train_Qagent2.size == 0 else np.vstack((self.dollar_train_Qagent2, Qagent2.dollar_train))
                    self.min_volt_Qagent2 = Qagent2.min_volt_test if self.min_volt_Qagent2.size == 0 else np.vstack((self.min_volt_Qagent2, Qagent2.min_volt_test))

                # train Q agent with true (unbiased) augmented data
                # prob_DNR.AssemblyQLearningData(gp, augmentdata='True augmented data')
                # Qagent3.Train(prob_DNR, num_steps=12000, batch_size=64, test_steps=self.test_steps, copy_steps=30)
                # self.dollar_Qagent3 = Qagent3.dollar_test if self.dollar_Qagent3.size == 0 else np.vstack((self.dollar_Qagent3, Qagent3.dollar_test))
                # self.dollar_train_Qagent3 = Qagent3.dollar_train if self.dollar_train_Qagent3.size == 0 else np.vstack((self.dollar_train_Qagent3, Qagent3.dollar_train))
                # self.min_volt_Qagent3 = Qagent3.min_volt_test if self.min_volt_Qagent3.size == 0 else np.vstack((self.min_volt_Qagent3, Qagent3.min_volt_test))

        self.dollar_historical = prob_DNR.total_money_ori
        self.dollar_optimal = prob_DNR.total_money_opt
        self.min_volt_historical = prob_DNR.minimum_volt_ori
        self.min_volt_optimal = prob_DNR.minimum_volt_opt

        # save the data
        summary_QL_dict = {
            "dollar_Qagent1": self.dollar_Qagent1,
            "dollar_Qagent2": self.dollar_Qagent2,
            #"dollar_Qagent3": self.dollar_Qagent3,
            "dollar_train_Qagent1": self.dollar_train_Qagent1,
            "dollar_train_Qagent2": self.dollar_train_Qagent2,
            #"dollar_train_Qagent3": self.dollar_train_Qagent3,
            "min_volt_Qagent1": self.min_volt_Qagent1,
            "min_volt_Qagent2": self.min_volt_Qagent2,
            #"min_volt_Qagent3": self.min_volt_Qagent3,
            "dollar_historical": self.dollar_historical,
            "dollar_optimal": self.dollar_optimal,
            "min_volt_historical": self.min_volt_historical,
            "min_volt_optimal": self.min_volt_optimal,
            "test_steps": self.test_steps
        }
        savemat("QLearningDataAugmentation.mat", summary_QL_dict)

    def plots(self):
        summary_QL_dict = loadmat("QLearningDataAugmentation.mat")
        dollar_Qagent1 = summary_QL_dict["dollar_Qagent1"]
        dollar_Qagent2 = summary_QL_dict["dollar_Qagent2"]
        #dollar_Qagent3 = summary_QL_dict["dollar_Qagent3"]
        min_volt_Qagent1 = summary_QL_dict["min_volt_Qagent1"]
        min_volt_Qagent2 = summary_QL_dict["min_volt_Qagent2"]
        #min_volt_Qagent3 = summary_QL_dict["min_volt_Qagent3"]
        dollar_historical = summary_QL_dict["dollar_historical"]
        dollar_optimal = summary_QL_dict["dollar_optimal"]
        min_volt_historical = summary_QL_dict["min_volt_historical"]
        min_volt_optimal = summary_QL_dict["min_volt_optimal"]
        test_steps = summary_QL_dict["test_steps"]
        if self.method == 'DQN':
            dollarQagent1_mean = np.mean(dollar_Qagent1, axis=0)
            dollarQagent1_10pt = np.percentile(dollar_Qagent1, 10, axis=0)
            dollarQagent1_90pt = np.percentile(dollar_Qagent1, 90, axis=0)
        else:
            dollarQagent2_mean = np.mean(dollar_Qagent2, axis=0)
            dollarQagent2_10pt = np.percentile(dollar_Qagent2, 10, axis=0)
            dollarQagent2_90pt = np.percentile(dollar_Qagent2, 90, axis=0)
        #dollarQagent3_mean = np.mean(dollar_Qagent3, axis=0)
       # dollarQagent3_10pt = np.percentile(dollar_Qagent3, 10, axis=0)
        #dollarQagent3_90pt = np.percentile(dollar_Qagent3, 90, axis=0)

        if self.method == 'DQN':
            minvoltQagent1_mean = np.mean(min_volt_Qagent1, axis=0)
            minvoltQagent1_10pt = np.percentile(min_volt_Qagent1, 10, axis=0)
            minvoltQagent1_90pt = np.percentile(min_volt_Qagent1, 90, axis=0)
            dollar_historical = np.squeeze(np.squeeze(dollar_historical * np.ones(shape=dollarQagent1_mean.shape)))
            dollar_optimal = np.squeeze(np.squeeze(dollar_optimal * np.ones(shape=dollarQagent1_mean.shape)))
            min_volt_historical = np.squeeze(np.squeeze(min_volt_historical * np.ones(shape=dollarQagent1_mean.shape)))
            min_volt_optimal = np.squeeze(np.squeeze(min_volt_optimal * np.ones(shape=dollarQagent1_mean.shape)))
        else:
            minvoltQagent2_mean = np.mean(min_volt_Qagent2, axis=0)
            minvoltQagent2_10pt = np.percentile(min_volt_Qagent2, 10, axis=0)
            minvoltQagent2_90pt = np.percentile(min_volt_Qagent2, 90, axis=0)
            dollar_historical = np.squeeze(np.squeeze(dollar_historical * np.ones(shape=dollarQagent2_mean.shape)))
            dollar_optimal = np.squeeze(np.squeeze(dollar_optimal * np.ones(shape=dollarQagent2_mean.shape)))
            min_volt_historical = np.squeeze(np.squeeze(min_volt_historical * np.ones(shape=dollarQagent2_mean.shape)))
            min_volt_optimal = np.squeeze(np.squeeze(min_volt_optimal * np.ones(shape=dollarQagent2_mean.shape)))
        #minvoltQagent3_mean = np.mean(min_volt_Qagent3, axis=0)
        #minvoltQagent3_10pt = np.percentile(min_volt_Qagent3, 10, axis=0)
        #minvoltQagent3_90pt = np.percentile(min_volt_Qagent3, 90, axis=0)

        # Plotly figure 1
        plotly.tools.set_credentials_file(username='yuanqigao', api_key='r9FQdeoRCUOMcw7zafs7')
        if self.method =='DQN':
            upper_bound_Qagent1 = go.Scatter(
                name='QL 90 percentile',
                showlegend=False,
                x=np.arange(0, test_steps * (dollarQagent1_90pt.size), test_steps),
                y=dollarQagent1_90pt,
                mode='lines',
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                fillcolor='rgba(0, 128, 255, 0.2)',  # color of upper shadow region
                fill='tonexty')
            trace_Qagent1 = go.Scatter(
                name='QL',
                x=np.arange(0, test_steps * (dollarQagent1_mean.size), test_steps),
                y=dollarQagent1_mean,
                mode='lines',
                line=dict(color='rgba(0, 128, 255, 1)'),  # color of average line
                fillcolor='rgba(0, 128, 255, 0.2)',  # color of lower shadow region
                fill='tonexty')
            lower_bound_Qagent1 = go.Scatter(
                name='QL 10 percentile',
                showlegend=False,
                x=np.arange(0, test_steps * (dollarQagent1_10pt.size), test_steps),
                y=dollarQagent1_10pt,
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                mode='lines')
        else:
            upper_bound_Qagent2 = go.Scatter(
                name='QL 90 percentile (with data augmentation)',
                showlegend=False,
                x=np.arange(0, test_steps * (dollarQagent2_90pt.size), test_steps),
                y=dollarQagent2_90pt,
                mode='lines',
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                fillcolor='rgba(255, 128, 0, 0.2)',  # color of upper shadow region
                fill='tonexty')
            trace_Qagent2 = go.Scatter(
                name='QL (with data augmentation)',
                x=np.arange(0, test_steps * (dollarQagent2_mean.size), test_steps),
                y=dollarQagent2_mean,
                mode='lines',
                line=dict(color='rgba(255, 128, 0, 1)'),  # color of average line
                fillcolor='rgba(255, 128, 0, 0.2)',  # color of lower shadow region
                fill='tonexty')
            lower_bound_Qagent2 = go.Scatter(
                name='QL 10 percentile (with data augmentation)',
                showlegend=False,
                x=np.arange(0, test_steps * (dollarQagent2_10pt.size), test_steps),
                y=dollarQagent2_10pt,
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                mode='lines')

        # upper_bound_Qagent3 = go.Scatter(
        #     name='QL 90 percentile (with true data augmentation)',
        #     showlegend=False,
        #     x=np.arange(0, test_steps * (dollarQagent3_90pt.size), test_steps),
        #     y=dollarQagent3_90pt,
        #     mode='lines',
        #     marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
        #     line=dict(width=0),
        #     fillcolor='rgba(0, 153, 76, 0.2)',  # color of upper shadow region
        #     fill='tonexty')
        # trace_Qagent3 = go.Scatter(
        #     name='QL (with true data augmentation)',
        #     x=np.arange(0, test_steps * (dollarQagent3_mean.size), test_steps),
        #     y=dollarQagent3_mean,
        #     mode='lines',
        #     line=dict(color='rgba(0, 153, 76, 1)'),  # color of average line
        #     fillcolor='rgba(0, 153, 76, 0.2)',  # color of lower shadow region
        #     fill='tonexty')
        # lower_bound_Qagent3 = go.Scatter(
        #     name='QL 10 percentile (with true data augmentation)',
        #     showlegend=False,
        #     x=np.arange(0, test_steps * (dollarQagent3_10pt.size), test_steps),
        #     y=dollarQagent3_10pt,
        #     marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
        #     line=dict(width=0),
        #     mode='lines')

        trace_ori = go.Scatter(
            name='Historical data',
            x=np.arange(0, test_steps * (dollar_historical.size), test_steps),
            y=dollar_historical,
            mode='lines',
            line=dict(color='rgba(153, 102, 255, 1)'),  # color of average line
            fill=None)

        trace_opt = go.Scatter(
            name='Optimal',
            x=np.arange(0, test_steps * (dollar_optimal.size), test_steps),
            y=dollar_optimal,
            mode='lines',
            line=dict(color='rgba(90, 90, 90, 1)'),  # color of average line
            fill=None)
        if self.method == 'DQN':
            data = [lower_bound_Qagent1, trace_Qagent1, upper_bound_Qagent1,
                    #lower_bound_Qagent2, trace_Qagent2, upper_bound_Qagent2,
                    #lower_bound_Qagent3, trace_Qagent3, upper_bound_Qagent3,
                    trace_ori,
                    trace_opt]
        else:
            data = [#lower_bound_Qagent1, trace_Qagent1, upper_bound_Qagent1,
                    lower_bound_Qagent2, trace_Qagent2, upper_bound_Qagent2,
                    # lower_bound_Qagent3, trace_Qagent3, upper_bound_Qagent3,
                    trace_ori,
                    trace_opt]

        layout = go.Layout(
            yaxis=dict(title='Total operational cost on testing week ($)'),
            xaxis=dict(title='Training iterations'),
            showlegend=True)

        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename='RL_reconfiguration.html')

        # Plotly figure 2
        plotly.tools.set_credentials_file(username='yuanqigao', api_key='r9FQdeoRCUOMcw7zafs7')
        if self.method == 'DQN':
            upper_bound_Qagent1 = go.Scatter(
                name='QL 90 percentile',
                showlegend=False,
                x=np.arange(0, test_steps * (minvoltQagent1_90pt.size), test_steps),
                y=minvoltQagent1_90pt,
                mode='lines',
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                fillcolor='rgba(0, 128, 255, 0.2)',  # color of upper shadow region
                fill='tonexty')
            trace_Qagent1 = go.Scatter(
                name='QL',
                x=np.arange(0, test_steps * (minvoltQagent1_mean.size), test_steps),
                y=minvoltQagent1_mean,
                mode='lines',
                line=dict(color='rgba(0, 128, 255, 1)'),  # color of average line
                fillcolor='rgba(0, 128, 255, 0.2)',  # color of lower shadow region
                fill='tonexty')
            lower_bound_Qagent1 = go.Scatter(
                name='QL 10 percentile',
                showlegend=False,
                x=np.arange(0, test_steps * (minvoltQagent1_10pt.size), test_steps),
                y=minvoltQagent1_10pt,
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                mode='lines')
        else:
            upper_bound_Qagent2 = go.Scatter(
                name='QL 90 percentile (with data augmentation)',
                showlegend=False,
                x=np.arange(0, test_steps * (minvoltQagent2_90pt.size), test_steps),
                y=minvoltQagent2_90pt,
                mode='lines',
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                fillcolor='rgba(255, 128, 0, 0.2)',  # color of upper shadow region
                fill='tonexty')
            trace_Qagent2 = go.Scatter(
                name='QL (with data augmentation)',
                x=np.arange(0, test_steps * (minvoltQagent2_mean.size), test_steps),
                y=minvoltQagent2_mean,
                mode='lines',
                line=dict(color='rgba(255, 128, 0, 1)'),  # color of average line
                fillcolor='rgba(255, 128, 0, 0.2)',  # color of lower shadow region
                fill='tonexty')
            lower_bound_Qagent2 = go.Scatter(
                name='QL 10 percentile (with data augmentation)',
                showlegend=False,
                x=np.arange(0, test_steps * (minvoltQagent2_10pt.size), test_steps),
                y=minvoltQagent2_10pt,
                marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
                line=dict(width=0),
                mode='lines')

        # upper_bound_Qagent3 = go.Scatter(
        #     name='QL 90 percentile (with true data augmentation)',
        #     showlegend=False,
        #     x=np.arange(0, test_steps * (minvoltQagent3_90pt.size), test_steps),
        #     y=minvoltQagent3_90pt,
        #     mode='lines',
        #     marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
        #     line=dict(width=0),
        #     fillcolor='rgba(0, 153, 76, 0.2)',  # color of upper shadow region
        #     fill='tonexty')
        # trace_Qagent3 = go.Scatter(
        #     name='QL (with true data augmentation)',
        #     x=np.arange(0, test_steps * (minvoltQagent3_mean.size), test_steps),
        #     y=minvoltQagent3_mean,
        #     mode='lines',
        #     line=dict(color='rgba(0, 153, 76, 1)'),  # color of average line
        #     fillcolor='rgba(0, 153, 76, 0.2)',  # color of lower shadow region
        #     fill='tonexty')
        # lower_bound_Qagent3 = go.Scatter(
        #     name='QL 10 percentile (with true data augmentation)',
        #     showlegend=False,
        #     x=np.arange(0, test_steps * (minvoltQagent3_10pt.size), test_steps),
        #     y=minvoltQagent3_10pt,
        #     marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
        #     line=dict(width=0),
        #     mode='lines')

        trace_ori = go.Scatter(
            name='Historical data',
            x=np.arange(0, test_steps * (min_volt_historical.size), test_steps),
            y=min_volt_historical,
            mode='lines',
            line=dict(color='rgba(153, 102, 255, 1)'),  # color of average line
            fill=None)

        trace_opt = go.Scatter(
            name='Optimal',
            x=np.arange(0, test_steps * (min_volt_optimal.size), test_steps),
            y=min_volt_optimal,
            mode='lines',
            line=dict(color='rgba(90, 90, 90, 1)'),  # color of average line
            fill=None)

        if self.method == 'DQN':
            data = [lower_bound_Qagent1, trace_Qagent1, upper_bound_Qagent1,
                    #lower_bound_Qagent2, trace_Qagent2, upper_bound_Qagent2,
                    #lower_bound_Qagent3, trace_Qagent3, upper_bound_Qagent3,
                    trace_ori,
                    trace_opt]
        else:
            data = [#lower_bound_Qagent1, trace_Qagent1, upper_bound_Qagent1,
                    lower_bound_Qagent2, trace_Qagent2, upper_bound_Qagent2,
                    # lower_bound_Qagent3, trace_Qagent3, upper_bound_Qagent3,
                    trace_ori,
                    trace_opt]
        # data = [lower_bound_Qagent1, trace_Qagent1, upper_bound_Qagent1,
        #         lower_bound_Qagent2, trace_Qagent2, upper_bound_Qagent2,
        #         lower_bound_Qagent3, trace_Qagent3, upper_bound_Qagent3,
        #         trace_ori,
        #         trace_opt]

        layout = go.Layout(
            yaxis=dict(title='Minimum voltage on testing week (p.u.)'),
            xaxis=dict(title='Training iterations'),
            #title='Performance over training iterations',
            showlegend=True)

        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename='RL_reconfiguration_volt.html')
        #plotly.io.orca.config.executable = '/home/yuanqi/Downloads/node-v10.15.3-linux-x64/lib/node_modules'
        #pio.write_image(fig, 'Q_performance.svg')

        fig_cnt = self.fig_cnt
        return fig_cnt
    # def plots_train(self):
    #     summary_QL_dict = loadmat("QLearningDataAugmentation.mat")
    #     dollar_train_Qagent1 = summary_QL_dict["dollar_train_Qagent1"]
    #     dollar_train_Qagent2 = summary_QL_dict["dollar_train_Qagent2"]
    #     dollar_train_Qagent3 = summary_QL_dict["dollar_train_Qagent3"]
    #     test_steps = summary_QL_dict["test_steps"]
    #
    #     dollarQagent1_mean = np.mean(dollar_train_Qagent1, axis=0)
    #     dollarQagent1_10pt = np.percentile(dollar_train_Qagent1, 10, axis=0)
    #     dollarQagent1_90pt = np.percentile(dollar_train_Qagent1, 90, axis=0)
    #     dollarQagent2_mean = np.mean(dollar_train_Qagent2, axis=0)
    #     dollarQagent2_10pt = np.percentile(dollar_train_Qagent2, 10, axis=0)
    #     dollarQagent2_90pt = np.percentile(dollar_train_Qagent2, 90, axis=0)
    #     dollarQagent3_mean = np.mean(dollar_train_Qagent3, axis=0)
    #     dollarQagent3_10pt = np.percentile(dollar_train_Qagent3, 10, axis=0)
    #     dollarQagent3_90pt = np.percentile(dollar_train_Qagent3, 90, axis=0)
    #
    #     # Plotly figure 1
    #     plotly.tools.set_credentials_file(username='yuanqigao', api_key='r9FQdeoRCUOMcw7zafs7')
    #
    #     upper_bound_Qagent1 = go.Scatter(
    #         name='QL 90 percentile',
    #         showlegend=False,
    #         x=np.arange(0, test_steps * (dollarQagent1_90pt.size), test_steps),
    #         y=dollarQagent1_90pt,
    #         mode='lines',
    #         marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
    #         line=dict(width=0),
    #         fillcolor='rgba(0, 128, 255, 0.2)',  # color of upper shadow region
    #         fill='tonexty')
    #     trace_Qagent1 = go.Scatter(
    #         name='QL',
    #         x=np.arange(0, test_steps * (dollarQagent1_mean.size), test_steps),
    #         y=dollarQagent1_mean,
    #         mode='lines',
    #         line=dict(color='rgba(0, 128, 255, 1)'),  # color of average line
    #         fillcolor='rgba(0, 128, 255, 0.2)',  # color of lower shadow region
    #         fill='tonexty')
    #     lower_bound_Qagent1 = go.Scatter(
    #         name='QL 10 percentile',
    #         showlegend=False,
    #         x=np.arange(0, test_steps * (dollarQagent1_10pt.size), test_steps),
    #         y=dollarQagent1_10pt,
    #         marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
    #         line=dict(width=0),
    #         mode='lines')
    #
    #     upper_bound_Qagent2 = go.Scatter(
    #         name='QL 90 percentile (with data augmentation)',
    #         showlegend=False,
    #         x=np.arange(0, test_steps * (dollarQagent2_90pt.size), test_steps),
    #         y=dollarQagent2_90pt,
    #         mode='lines',
    #         marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
    #         line=dict(width=0),
    #         fillcolor='rgba(255, 128, 0, 0.2)',  # color of upper shadow region
    #         fill='tonexty')
    #     trace_Qagent2 = go.Scatter(
    #         name='QL (with data augmentation)',
    #         x=np.arange(0, test_steps * (dollarQagent2_mean.size), test_steps),
    #         y=dollarQagent2_mean,
    #         mode='lines',
    #         line=dict(color='rgba(255, 128, 0, 1)'),  # color of average line
    #         fillcolor='rgba(255, 128, 0, 0.2)',  # color of lower shadow region
    #         fill='tonexty')
    #     lower_bound_Qagent2 = go.Scatter(
    #         name='QL 10 percentile (with data augmentation)',
    #         showlegend=False,
    #         x=np.arange(0, test_steps * (dollarQagent2_10pt.size), test_steps),
    #         y=dollarQagent2_10pt,
    #         marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
    #         line=dict(width=0),
    #         mode='lines')
    #
    #     upper_bound_Qagent3 = go.Scatter(
    #         name='QL 90 percentile (with true data augmentation)',
    #         showlegend=False,
    #         x=np.arange(0, test_steps * (dollarQagent3_90pt.size), test_steps),
    #         y=dollarQagent3_90pt,
    #         mode='lines',
    #         marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
    #         line=dict(width=0),
    #         fillcolor='rgba(0, 153, 76, 0.2)',  # color of upper shadow region
    #         fill='tonexty')
    #     trace_Qagent3 = go.Scatter(
    #         name='QL (with true data augmentation)',
    #         x=np.arange(0, test_steps * (dollarQagent3_mean.size), test_steps),
    #         y=dollarQagent3_mean,
    #         mode='lines',
    #         line=dict(color='rgba(0, 153, 76, 1)'),  # color of average line
    #         fillcolor='rgba(0, 153, 76, 0.2)',  # color of lower shadow region
    #         fill='tonexty')
    #     lower_bound_Qagent3 = go.Scatter(
    #         name='QL 10 percentile (with true data augmentation)',
    #         showlegend=False,
    #         x=np.arange(0, test_steps * (dollarQagent3_10pt.size), test_steps),
    #         y=dollarQagent3_10pt,
    #         marker=dict(color="#444"),  # if line=dict(width=0), then this color is useless
    #         line=dict(width=0),
    #         mode='lines')
    #     data = [lower_bound_Qagent1, trace_Qagent1, upper_bound_Qagent1,
    #             lower_bound_Qagent2, trace_Qagent2, upper_bound_Qagent2,
    #             lower_bound_Qagent3, trace_Qagent3, upper_bound_Qagent3]
    #
    #     layout = go.Layout(
    #         yaxis=dict(title='Total operational cost on training week ($)'),
    #         xaxis=dict(title='Training iterations'),
    #         showlegend=True)
    #
    #     fig = go.Figure(data=data, layout=layout)
    #     plotly.offline.plot(fig, filename='RL_reconfiguration_train.html')




if __name__ == "__main__":
    # method = sys.argv[1]
    # print(method)
    exp1 = Experiment1(test_steps=100, method='DQN')
    exp1.run(num_repeats=10, np_seed=10, tf_seed=1)
    fig_cnt = exp1.plots()
    DEBUGGER_ROAD_BLOCK = 'DEBUG MODE: BLOCK PROGRAM HERE'
    input()

# IMPORTANT CHANGES TO THE WORKING VERSION:
# (1). the sign of the Lagrangian constraint term is corrected. In main_DQN_historical_data.py it was +1, but we realize
#      in this file that it should be -1
# (2). the augmented data can only select the configurations that appeared in the historical data set. You cannot
#      randomly select from the 83 configurations
