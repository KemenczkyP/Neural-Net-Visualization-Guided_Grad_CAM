# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:14:11 2019

@author: KemenczkyP
"""

import tensorflow as tf

class CamModel(tf.keras.Model):

    def __init__(self, input_size = (1,1)):
        super(CamModel, self).__init__()
        self.gap_layer = tf.keras.layers.AveragePooling2D(input_size,
                                            padding= 'valid',
                                            strides = (1,1))

    def call(self, x):
        x = self.gap_layer(x)
        return x