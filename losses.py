import numpy as np

from torch.autograd import Variable
from skimage.feature import daisy
from torch import nn
import torch
from torch import autograd
from utils import *


class WGANLoss(torch.nn.Module):
    '''
        WGAN loss as written in
        https://github.com/github-pengge/PyTorch-progressive_growing_of_gans.git
    '''

    def __init__(self):
        super(WGANLoss, self).__init__()

    def forward(self,predicted,true):
        WGAN = (-2*true+1) * torch.mean(predicted)
        return WGAN

def R1_reg(dloss_real,d_real,x_real,reg_param=10):
    '''
        R1 regularization term from
        https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
        based on the paper: "Which Training Methods for GANs do actually Converge?"
        --> "Our analysis suggests that the main effect of the zerocentered
            gradient penalties proposed by Roth et al. (2017)
            on local stability is to penalize the discriminator for deviating
            from the Nash-equilibrium. The simplest way to achieve
            this is to penalize the gradient on real data alone: when the
            generator distribution produces the true data distribution
            and the discriminator is equal to 0 on the data manifold, the
            gradient penalty ensures that the discriminator cannot create
            a non-zero gradient orthogonal to the data manifold without
            suffering a loss in the GAN game".
    '''
    # dloss_real.backward(retain_graph=True)
    # reg = sum([reg_param * compute_grad2(d_real, x_real).mean()])
    # reg.backward()
    d_adv_loss_real(retain_graph=True)
    d_real_reg = sum([reg_param * compute_grad2(d_real_i, Img_real_i).mean() for d_real_i,Img_real_i in zip(d_real,x_real)])
    d_real_reg.backward(retain_graph=False)


def R2_reg(dloss_fake,d_fake,x_fake,reg_param=10):
    '''
        R1 regularization term from
        https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
        based on the paper: "Which Training Methods for GANs do actually Converge?"
        --> "We penalize the discriminator gradients on the current
            generator distribution instead of the true data distribution."
    '''
    dloss_fake.backward(retain_graph=True)
    reg = reg_param * compute_grad2(d_fake, x_fake).mean()
    reg.backward()

def compute_grad2(d_out, x_in):
    '''
        Output the squared gradient of d_out wrt x_in
    '''
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
