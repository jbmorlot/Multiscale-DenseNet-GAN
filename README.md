# Multiscale-DenseNet-GAN
(under developement)
![MSDN-GAN](https://github.com/jbmorlot/Multiscale-DenseNet-GAN/blob/master/images/Diagram.png)

Recent papers has shown that adding dense connection between convolutional layers on a samll model perform better and train faster than large networks as ResNet ([Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)). An improvement suggests to add top-down (or downsampling) links to improve the classification score ([Multi-Scale Dense Networks for Resource Efficient Image Classification
](https://arxiv.org/abs/1703.09844)). 
 
This network aims to generate an efficient embedding at every scales using Multiscale DenseNet and GAN frameworks (see Figure). The generator from the GAN uses features discovered at each scales to fool the discriminator and the discriminator uses the same architecture to define at each scale if the image has been genearted or not. 

The GAN uses ([Wasserstein gan loss](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)) and a regularization minimizing the weight of the links. 



