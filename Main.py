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

import cv2 as cv

print(tf.__version__)

from kerasmodel import MyModel
from heatmap import HeatMap

#register Guided ReLU function
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
   gate_f = tf.cast(op.outputs[0] > 0, "float32") #for f^l > 0
   gate_R = tf.cast(grad > 0, "float32") #for R^l+1 > 0
   return gate_f * gate_R * grad
#%%
# Load dataset

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

# scale data to [0,1]
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)
#%%

#TRAIN
with tf.compat.v1.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}): #use Guided ReLU
    model = MyModel()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    
    model.fit(train_images, train_labels, epochs=1,
              batch_size = 100)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
#%%
    
#COMPUTE GRADIENTS AND VISUALIZE 
#source: 
#   Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
#   doi: 10.1109/iccv.2017.74
    
    random_sample = np.random.randint(0, test_images.shape[0], 1)[0]
    image = test_images[random_sample:random_sample+1,:,:] # 
    image = tf.convert_to_tensor(image)
    label = test_labels[random_sample:random_sample+1]
    
    im = tf.convert_to_tensor(np.reshape(image,(-1,image.shape[1], image.shape[2],1)), tf.float32)
    with tf.GradientTape() as logits_tape, tf.GradientTape() as loss_tape:
        
        loss_tape.watch(image)
        
        logits = model(image)
        loss_value = loss_fn(label, logits)
        
        lastc = model.last_conv_value
        
    guide = loss_tape.gradient(loss_value, image)
    grads = logits_tape.gradient(logits, lastc)

    GAP_pool = tf.keras.layers.AveragePooling2D((lastc.shape[1], lastc.shape[2]),
                                                padding= 'valid',
                                                strides = (1,1))(grads)
    
    grad_c=tf.zeros(lastc.shape[1:3], tf.float32)
    for idx in range(0,GAP_pool.shape[3]):
        grad_c=tf.nn.relu(grad_c+lastc[0,:,:,idx]*GAP_pool[0,:,:,idx])

grad_cam, heatmap = HeatMap(grad_c,guide, dims = 2)

#show with heatmap
image = image.numpy() * 255 #rescale to original
image = np.squeeze(np.uint8(image))

RGB_img = cv.cvtColor(image, cv.COLOR_GRAY2BGR) #convert to "RGB" (size)

heatmap_img = cv.applyColorMap(np.uint8(heatmap), cv.COLORMAP_JET)

fin = cv.addWeighted(heatmap_img, 0.7, RGB_img, 0.3, 0)

plt.imshow(fin)
#cv.imshow('image_w_heatmap', fin)
