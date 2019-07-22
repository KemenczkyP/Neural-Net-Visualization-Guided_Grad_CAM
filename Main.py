# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:46:40 2019

@author: KemenczkyP
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

from kerasmodel import MyModel
from cammodel import CamModel

@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
   gate_f = tf.cast(op.outputs[0] > 0, "float32") #for f^l > 0
   gate_R = tf.cast(grad > 0, "float32") #for R^l+1 > 0
   return gate_f * gate_R * grad
#%%
# READ LIVS

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']

train_images = (train_images / 255.0).astype(np.float32)

test_images = (test_images / 255.0).astype(np.float32)
#%%

with tf.compat.v1.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
    model = MyModel()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    
    model.fit(train_images, train_labels, epochs=5,
              batch_size = 100)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    #%%
    

with tf.compat.v1.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
    image = test_images[10:11,:,:]
    label = test_labels[10:11]
    with tf.GradientTape(persistent=True) as tape:
        logits = model(image)
        loss_value = loss_fn(label, logits)
        
        lastc = model.last_conv_value
        
    guide = tape.gradient(loss_value,
                              tf.convert_to_tensor(np.reshape(image,(-1,image.shape[1], image.shape[2],1)), tf.float32))
    grads = tape.gradient(logits, lastc)
    
    GAP_pool = tf.keras.layers.AveragePooling2D((lastc.shape[1], lastc.shape[2]),
                                                padding= 'valid',
                                            strides = (1,1))(grads)
    
    grad_c=tf.zeros(lastc.shape[1:3], tf.float32)
    for idx in range(0,GAP_pool.shape[3]):
        grad_c=tf.nn.relu(grad_c+lastc[0,:,:,idx]*GAP_pool[0,:,:,idx])
    plt.imshow(image[0,:,:])
    plt.imshow(grad_c)
    
    
