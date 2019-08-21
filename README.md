# Neural-Net-Visualization

Guided Grad-Cam visualization based on the article "Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization"(doi:10.1109/iccv.2017.74). 
Implemeted in Tensorflow 2 ('2.0.0-beta1') with keras. 
Rarely the gradient computing fails (between logits and the feature map after the last convolution) and generates zero heatmap.
