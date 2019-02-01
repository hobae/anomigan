from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout, Input
from keras.layers import Dense, Flatten, GlobalMaxPooling1D 
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

from sklearn.metrics import roc_curve, auc
from scipy import stats
from scipy.signal import butter, lfilter, freqz

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import KFold


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import random
import sys
import os

GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

class ANOMIGAN():
    def __init__(self):
        self.testfile = #TEST_FILE 
        self.data = #TRAIN_FILE
        self.num_feature = 30
        self.X_test = 0 
        self.X_gen = 0 
        self.scaler = MaxAbsScaler() 
        self.input_shape = (-1,-1) 
        self.latent_dim = 100
        self.C= tf.placeholder(tf.float32, [None, 512])
        self.C_prime = tf.placeholder(tf.float32, [None, 512])

        # models
        self.generator = None
        self.discriminator = None
        self.preTrainedModel = Sequential()

        # hyperparameter for loss
        self.lambda_a = 0.5
        self.lambda_b = 1 - self.lambda_a
        self.confidence = 1.0
        self.batch_size = 32
        self.num_variance = 5

        # temp Lists
        self.bList = []
        self.aList = []

        self.bFpr = []
        self.bTpr = []
        self.bThresholds = []

        self.aFpr = []
        self.aTpr = []
        self.aThresholds = []

        self.t_var = {}

######## drawring functions ###############
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def drawLoss(self, S_loss_list, E_loss_list):
        # Filter requirements.
        order = 6
        fs = 30.0       # sample rate, Hz
        cutoff = 3.667  # desired cutoff frequency of the filter, Hz

        s_filter = self.butter_lowpass_filter(S_loss_list, cutoff, fs, order)
        d_filter = self.butter_lowpass_filter(E_loss_list, cutoff, fs, order)

        ylim = [0,3]
        f = plt.figure(tight_layout=True)  
        ax = f.add_subplot(111, ylim=ylim)
        ax.set_xlabel("Epochs",fontsize=20)
        ax.set_ylabel("Loss",fontsize=20)
        ax.plot(s_filter, label='Discriminator', color='blue', linewidth=1, linestyle='--' )
        ax.plot(d_filter, label='Encoder', color='green', linewidth=1, alpha=0.5 )
        ax.legend(loc=1,fontsize=15)

        plt.show()

    def drawAccuracyPlot(self):
        ylim = [0,105]
        xlim = [0, 10]
        f = plt.figure(tight_layout=True)  
        ax = f.add_subplot(111, ylim=ylim)
        ax.set_xlabel("Random Iterative Steps",fontsize=20)
        ax.set_ylabel("Accuracy",fontsize=20)
        plt.plot(self.bList, label='Original Samples', color='blue', linewidth=1, linestyle='--' )
        plt.plot(self.aList, label='Generated Samples', color='green', linewidth=1, )
        plt.legend()

        plt.show()

    def drawRocPlot(self):
        fpr1, tpr1, thresholds1 = self.bFpr, self.bTpr, self.bThresholds   
        roc_auc1 = auc(fpr1, tpr1)
 
        fpr2, tpr2, thresholds2 = self.aFpr, self.aTpr, self.aThresholds
        roc_auc2 = auc(fpr2, tpr2)

        plt.figure()
        plt.plot(fpr1, tpr1, color='blue', linestyle='--', linewidth=2, label='ROC curve with original samples (area = %0.2f)' % roc_auc1)
        plt.plot(fpr2, tpr2, color='green', linewidth=1, label='ROC curve with generated samples (area = %0.2f)' % roc_auc2)
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

######## target classifer model functions ###############
    def get_pretrainModel(self):
        #self.preTrainedModel ; # USE API to get target pretrained Model

    def get_target_features(self):
        X = 1 # Define input features of X
        Y = 1 # Define label of input features of X
        return X, Y 

######## AnomiGAN model functions ###############
    def discriminator(self, x):
        with tf.variable_scope("discriminator"):
            x_reshaped = tf.reshape(x, (-1, self.num_feature, 1))
            conv1 = tf.layers.conv1d(x_reshaped, filters=32, kernel_size=4, 
                                                        strides=2,
                                                        padding='VALID',
                                                        activation=tf.nn.relu)
            conv2 = tf.layers.conv1d(conv1, filters=10,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)
            conv3 = tf.layers.conv1d(conv1, filters=20,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)
            conv4 = tf.layers.conv1d(conv1, filters=30,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)
            flatten = tf.layers.flatten(conv4)
            out = tf.layers.dense(flatten, self.num_feature, activation=tf.nn.relu)
            return out

    def operation_mode(self, x, message):
        if mode == 1:
            dtype = x.dtype
            x_btensor = tf.cast(x, tf.int32)
            m_btensor = tf.cast(message, tf.int32)
            xor = tf.bitwise.bitwise_xor(x_btensor, m_btensor)
            random = tf.cast(xor, dtype)
        else:
            random = x*message % np.amax(x) 

    def encoder(self, x, message, mode):
        with tf.variable_scope("encoder"):
            random = operation_mode(x, message, mode)
            x_flatten = tf.layers.flatten(random)
            fc1 = tf.reshape(x_flatten, (-1, self.num_feature, 1)) 
            conv1d_t1 = tf.layers.conv1d(fc1, filters=64, kernel_size=4, 
                                                        strides=2,
                                                        padding='VALID',
                                                        activation=tf.nn.relu)
            bn1 = tf.layers.batch_normalization(conv1d_t1)
            
            conv1d_t2 = tf.layers.conv1d(bn1, filters=32,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)

            bn2 = tf.layers.batch_normalization(conv1d_t2)

            conv1d_t3 = tf.layers.conv1d(bn2, filters=16,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)

            bn3 = tf.layers.batch_normalization(conv1d_t3)
            conv1d_t4 = tf.layers.conv1d(bn3, filters=8,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)

            bn4 = tf.layers.batch_normalization(conv1d_t4)
            conv1d_t5 = tf.layers.conv1d(bn4, filters=4,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)
    
            bn5 = tf.layers.batch_normalization(conv1d_t5)
            conv1d_t6 = tf.layers.conv1d(bn5, filters=8, 
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)
 
            bn6 = tf.layers.batch_normalization(conv1d_t6)
            conv1d_t7 = tf.layers.conv1d(bn6, filters=16, 
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)

            bn7 = tf.layers.batch_normalization(conv1d_t7)
            conv1d_t8 = tf.layers.conv1d(bn7, filters=self.num_feature, 
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation=tf.nn.tanh)
            flatten = tf.layers.flatten(conv1d_t8)
            out = tf.layers.dense(flatten, self.num_feature, activation=tf.nn.relu)
            return out

    def get_solvers(self, learning_rate=1e-3, beta1=0.5):
        E_solver = tf.train.AdamOptimizer(learning_rate, beta1)
        S_solver = tf.train.AdamOptimizer(learning_rate, beta1)
        return E_solver, S_solver

    def train(self, sess, E_train_step, S_train_step, E_loss, S_loss, epochs=3000, batch_size=10):
        X, Y = self.get_target_features() 
        for it in range(epochs):
            minibatch, labels = self.get_shuffle_batch(X, Y, batch_size)
            minibatch = minibatch.reshape(batch_size, -1)

            if epochs > (epochs - 2000): 
                self.store_parameters(sess)

            #randomize original data
            fake = np.random.normal(0, 1, (batch_size, 30))
            randomized = sess.run(self.C_prime, feed_dict = {self.C:minibatch, self.random:fake})
            loss = self.target_classifier(randomized, labels, batch_size) 

            _, S_loss_curr = sess.run([S_train_step, S_loss], feed_dict={self.C:minibatch, self.random:fake,
            self.loss:loss})

            _, E_loss_curr = sess.run([E_train_step, E_loss], feed_dict={self.C:minibatch, self.random:fake,
            self.loss:loss})

            S_loss_list.append(S_loss_curr)
            E_loss_list.append(np.mean(E_loss_curr))

        #self.drawLoss(S_loss_list, E_loss_list) 
        print ("Train Finishied")

    def target_classifier(self, fake, fake_label, batch_size=32):
        cvscores = []
        scores = self.preTrainedModel.evaluate(fake, fake_label, verbose=0)
        output = np.mean(scores[1])
        return output 

    def calculate_loss(self, C, C_prime, logit_real, logit_fake, loss):
        real_label = tf.ones_like(logit_real)
        fake_label = tf.zeros_like(logit_fake)

        loss_S_real = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=real_label, logits=logit_real)
        loss_S_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=fake_label, logits=logit_fake)

        loss_S = (tf.reduce_mean(loss_S_real) + (tf.reduce_mean(loss_S_fake)* (1-tf.reduce_mean(loss))))
        C_flatten = tf.layers.flatten(C)
        C_prime_flatten = tf.layers.flatten(C_prime)
        distance = (tf.sqrt(tf.reduce_sum(tf.square(C_flatten - C_prime_flatten), axis=1))) 
        distance = tf.reduce_mean(distance)
        loss_E = (self.lambda_a * (distance*self.confidence) )  + (self.lambda_b * loss_S)
        return loss_E, loss_S 

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        return session

    def get_shuffle_batch(self, X, Y, batch_size=32):
        idx = random.randint(1, len(X)-batch_size)
        return  X[idx:idx+batch_size], Y[idx:idx+batch_size]

    def get_next_batch(self, X, Y, start, end, batch_size=32):
        X_train = []
        Y_train = []
        start = 0
        end = batch_size
        for i in range(len(X)-batch_size):
            start+=i
            end+=i
            X_train.append(X[start:end])
            Y_train.append(Y[start:end])
        return X_train, Y_train 

    def anonymize_sample(self, sess, batch_size):
        minibatch, labels = self.get_target_features() 
        batch_size = len(Y)
        fake = np.random.normal(0, 1, (batch_size, 30))
        randomized = sess.run(self.C_prime, feed_dict = {self.C:minibatch, self.random:fake})
        scores1 = self.preTrainedModel.evaluate(randomized, Y, verbose=0)
        cvscores1.append(scores1[1] * 100)
        self.get_inversed(randomized) 

    def get_inversed(self, normalized):
        np.set_printoptions(precision=6, suppress=True)
        inversed = self.scaler.inverse_transform(normalized)
        np.savetxt('fileout.txt', inversed, delimiter=',', fmt='%1.3f') 
        return inversed

    def store_parameters(self, sess):
        for i in range(1, 7):
            name = 'Encoder/conv1d_' + i + '/kernel:0'
            conv = sess.graph.get_tensor_by_name(name)
            self.t_var[name] = conv
            self.t_var.append(name, sess.run(conv))

    def add_variance(self, sess, num_var):
        for i in range(num_var):
            num = random.randint(1, 7)
            name = 'Encoder/conv1d_' + num + '/kernel:0'
            conv = sess.graph.get_tensor_by_name(name)
            var = np.var(self.t_var.get(name)), axis=0)
            sess.run(tf.assign(conv, conv + var))

    def get_pvalue(self, a, b):
        a = a.flatten()
        b = b.flatten() 
        t, p = stats.pearsonr(a,b)

    def main(self):
        self.get_pretrainModel()

        tf.reset_default_graph()

        self.C = tf.placeholder(tf.float32, [None, self.num_feature])
        self.random = tf.placeholder(tf.float32, [None, self.num_feature])

        self.C_prime = self.encoder(self.C, self.random, mode=2)
        self.loss = tf.placeholder(tf.float32)

        with tf.variable_scope("") as scope:
            logit_real = self.discriminator(self.C)
            scope.reuse_variables()
            logit_fake = self.discriminator(self.C_prime)


        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
        steganalayzer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

        E_solver, S_solver = self.get_solvers()

        E_loss, S_loss = self.calculate_loss(self.C, self.C_prime, logit_real, logit_fake, self.loss)
        E_train_step = E_solver.minimize(E_loss, var_list=encoder_vars)
        S_train_step = E_solver.minimize(S_loss, var_list=steganalayzer_vars)

        tf.executing_eagerly()
        sess = self.get_session() 
        sess.run(tf.global_variables_initializer())
        self.train(sess, E_train_step, S_train_step, E_loss, S_loss)

        self.add_variance(sess, self.num_variance) 
        self.anonymize_sample(sess, self.batch_size)       


if __name__ == '__main__':
    anomigan = ANOMIGAN()
    anomigan.main()
