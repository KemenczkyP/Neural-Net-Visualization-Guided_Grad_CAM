# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:14:11 2019

@author: KemenczkyP
"""

import tensorflow as tf

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn1 = tf.keras.layers.Conv2D(4,
                                           [3,3],
                                           strides=(1, 1),
                                           padding='same',
                                           activation=tf.nn.leaky_relu)
        self.cnn2 = tf.keras.layers.Conv2D(16,
                                           [3,3],
                                           strides=(1, 1),
                                           padding='same',
                                           activation=tf.nn.leaky_relu)
        self.cnn3 = tf.keras.layers.Conv2D(32,
                                           [3,3],
                                           strides=(1, 1),
                                           padding='same',
                                           activation=tf.nn.leaky_relu)
    
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2,2),
                                                  padding = 'same',
                                                  strides = (1,1))

    def call(self, x):
        x = tf.keras.backend.reshape(x, (-1,28,28,1))
        x = self.cnn1(x)
        '''
        x = self.max_pool(x)
        x = self.cnn2(x)
        x = self.max_pool(x)
        x = self.cnn3(x) 
        '''
        self.last_conv_value = self.max_pool(x)
        x = tf.keras.backend.reshape(self.last_conv_value,
                                     (-1,
                                      self.last_conv_value.shape[1] * self.last_conv_value.shape[2] * self.last_conv_value.shape[3]))
        x = self.dense1(x)
        return self.dense2(x)