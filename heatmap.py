# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:10:41 2019

@author: KemenczkyP
"""

def HeatMap(grads, guided_bp, dims = 2):
    '''
    \n Makes a dims-D heatmap from computed gradients.
    \n -------------------------
    \n input: grads and guided_bp, the computed gradient
    \n dims: 2 for images, 3 for volumes
    \n output: grad_cam, heatmap
    '''
    import scipy.ndimage
    import numpy as np
    import matplotlib.pyplot as plt
        
    if(dims == 2):
        zoom_size = [guided_bp.shape[1]/grads.shape[0],
                     guided_bp.shape[2]/grads.shape[1]]
        
    elif(dims == 3):
        zoom_size = [guided_bp.shape[1]/grads.shape[0],
                     guided_bp.shape[2]/grads.shape[1],
                     guided_bp.shape[3]/grads.shape[2]]
        
    g_c=scipy.ndimage.zoom(grads, zoom_size)
       
    
    heatmap=np.multiply(g_c,
                        np.squeeze(guided_bp))
    minn=np.min(heatmap)
    heatmap=heatmap-minn
    maxx=np.max(heatmap)
    heatmap=heatmap/maxx*255
    
    return g_c,heatmap