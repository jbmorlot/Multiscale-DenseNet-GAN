import math
import numpy as np


import torch.nn.functional as F
from torch import nn
from torch.nn.init import kaiming_normal, calculate_gain
import torch

from utils import *

class MSGenerator(nn.Module):
    '''
        Multiscale Generator
        NF: Number of filters per blocks
    '''
    def __init__(self,NF,scales,depths,Ndepth_max,SlopLRelu,latent_dim,Adj):
        super(MSGenerator, self).__init__()

        self.Nscale = len(scales)
        self.Ndepth = len(depths)
        self.Ndepth_max = Ndepth_max

        self.Fromlatent = nn.ModuleList([FromLatent(NF,scale,SlopLRelu,latent_dim)
                                         for scale in scales])

        # self.Adj = nn.Parameter(torch.from_numpy(Adj))

        self.Block_table = nn.ModuleList(sum([[Blocks(NF,s,d,Ndepth_max,scales,SlopLRelu) for d in range(self.Ndepth)]
                                                for s in range(self.Nscale)],[])
                                         )

        # self.Up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.Down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ToImg = nn.ModuleList([ToImg(NFilter=NF,SlopLRelu=SlopLRelu) for _ in range(self.Nscale)])

    def forward(self,latent_codes,Adj):


        def get_scale(X,s,d,NUp=0,NDown=0):

            #print('(s;d) = ({0},{1})'.format(s,d))
            if d==0:
                return self.Block_table[s*self.Ndepth+d](X[s],NUp,NDown)

            Adj_layers = []
            for scale in range(self.scales):
                for depth in range(self.Ndepth):

                    if depth==d-1 and scale==s:
                        Adj_layers.append(get_scale(X,scale,depth)) #Add previous layer at fixed scale with weight 1

                    coupling = self.Adj[scale*self.Ndepth+depth,s*self.Ndepth+d]
                    if coupling>0:
                        #Adjusting layers scalings
                        NUp=0
                        NDown=0
                        N = scale-s
                        if N>0:
                            NDown=N
                        elif N<0:
                            NUp=-N

                        Adj_layers.append(coupling.expand_as(X[0])*get_scale(X,scale,depth,NUp,NDown))

            # print('before block (s;d) = ({0},{1})'.format(s,d))
            return self.Block_table[s*self.Ndepth+d](Adj_layers,NUp,NDown)

        #Generating latent code at each scale
        LatentImg = [None]*self.Nscale
        for s in range(self.Nscale):
            LatentImg[s] = self.Fromlatent[s](latent_codes[s])

        GenImg = [None]*self.Nscale
        #Pass throug the total network
        for s in range(self.Nscale):
            GenImg[s] = self.ToImg[s](get_scale(LatentImg,s,self.Ndepth-1))

        return GenImg


class Blocks(nn.Module):
    '''
        Blocks of the generator
    '''
    def __init__(self,NF,s,d,dmax,scales,SlopLRelu):
        super(Blocks, self).__init__()

        self.d = d
        self.s = s
        #Define the number of filters
        NFdepths = d if d<dmax else dmax
        if d>0:
            if s>0 and s<len(scales)-1:
                NFscales = 2
            else:
                NFscales = 1
        else:
            NFdepths = 1
            NFscales = 0



        NFilterIn = (NFdepths + NFscales)*NF
        NFilterOut = NF
        print('(NFilterIn) = ({0})'.format(NFilterIn))
        self.numFilt = NFilterIn
        self.NFdepths = NFdepths
        self.NFscales = NFscales

        # self.Conv1 = nn.Conv2d(in_channels=NFilterIn,out_channels=NFilterOut*4,kernel_size=1,padding=0)
        # self.ConvBN1 = nn.BatchNorm2d(NFilterIn)
        self.Conv3 = nn.Conv2d(in_channels=NFilterIn,out_channels=NFilterOut,kernel_size=3,padding=1)
        # self.Conv3 = nn.Conv2d(in_channels=NFilterOut*4,out_channels=NFilterOut,kernel_size=3,padding=1)
        self.ConvBN3 = nn.BatchNorm2d(NFilterOut)
        # self.ConvBN3 = nn.BatchNorm2d(4*NFilterOut)
        self.LRelu = nn.LeakyReLU(SlopLRelu)

        # self.Up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.Down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsampling = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                        nn.Conv2d(in_channels=NFilterOut,out_channels=NFilterOut,kernel_size=3,padding=1),
                                        nn.BatchNorm2d(NFilterOut),
                                        nn.LeakyReLU(SlopLRelu),
                                        )
        self.Downsampling = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(in_channels=NFilterOut,out_channels=NFilterOut,kernel_size=3,padding=1),
                                          nn.BatchNorm2d(NFilterOut),
                                          nn.LeakyReLU(SlopLRelu),
                                         )
    def forward(self,X,Up=False,Down=False):

        if type(X)==list:
            X = torch.cat(tuple(X),1)
        # print(X.size(),self.s,self.d,self.numFilt,self.NFscales,self.NFdepths)

        # X = self.ConvBN1(X)
        # X = self.LRelu(X)
        # X = self.Conv1(X)

        X = self.Conv3(X)
        X = self.ConvBN3(X)
        X = self.LRelu(X)

        if Up:
            X = self.Upsampling(X)

        if Down:
            X = self.Downsampling(X)

        return X

class FromLatent(nn.Module):
    def __init__(self,NF,scale,SlopLRelu,latent_dim):
        super(FromLatent, self).__init__()
        self.scale = scale
        self.NFilter = NF

        self.Linear = nn.Linear(latent_dim,self.NFilter*self.scale**2)
        # self.Conv1 = nn.Conv2d(in_channels=NFilter,out_channels=NFilter,kernel_size=5,padding=2)
        self.Conv3 = nn.Conv2d(in_channels=self.NFilter,out_channels=self.NFilter,kernel_size=3,padding=1)
        self.ConvBN = nn.BatchNorm2d(self.NFilter)
        self.LRelu = nn.LeakyReLU(SlopLRelu)

    def forward(self,X):

        X = self.Linear(X)
        X = X.view(X.shape[0], self.NFilter, self.scale, self.scale)
        X = self.ConvBN(X)
        X = self.LRelu(X)
        X = self.Conv3(X)

        return X

class ToImg(nn.Module):
    def __init__(self,NFilter,SlopLRelu):
        super(ToImg, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=NFilter,out_channels=NFilter,kernel_size=3,padding=1)
        self.Conv2 = nn.Conv2d(in_channels=NFilter,out_channels=3,kernel_size=1,padding=0)
        self.LRelu = nn.LeakyReLU(SlopLRelu)
        self.Relu = nn.ReLU()
        self.Sig = nn.Sigmoid()

    def forward(self,X):
        # Ximg={}
        X = self.Conv1(X)
        X = self.LRelu(X)

        X = self.Conv2(X)
        X = self.Sig(X)

        return X



class MSDiscriminator(nn.Module):
    '''
        Multiscale Discriminator
        NF: Number of filters per blocks
    '''
    def __init__(self,NF,scales,depths,Ndepth_max,SlopLRelu,Nfeatures):
        super(MSDiscriminator, self).__init__()

        self.Nscale = len(scales)
        self.Ndepth = len(depths)
        self.Ndepth_max = Ndepth_max

        self.FromImg = nn.ModuleList([FromImg(NF,SlopLRelu) for _ in range(self.Nscale)])

        self.Block_table = nn.ModuleList(sum([[Blocks(NF,s,d,Ndepth_max,scales,SlopLRelu) for d in range(self.Ndepth)]
                                                for s in range(self.Nscale)],[])
                                         )

        self.Up = nn.Upsample(scale_factor=2, mode='nearest')
        self.Down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ToProba = nn.ModuleList([ToProba(NFilter=NF,scale=scale,Nfeature=Nfeatures[s],SlopLRelu=SlopLRelu)
                                      for s,scale in enumerate(scales)])

    def forward(self,GenImg):

        def get_scale(X,s,d,Up=False,Down=False):

            #print('(s;d) = ({0},{1})'.format(s,d))
            if d==0:
                return self.Block_table[s*self.Ndepth+d](X[s],Up,Down)

            dmin = d-self.Ndepth_max if d-self.Ndepth_max>=0 else 0
            Depth_layers = [get_scale(X,s,dd) for dd in range(dmin,d)]

            Scale_layers = []
            if s<self.Nscale-1:
                Scale_layers = Scale_layers + [get_scale(X,s+1,d-1,Down=True)]

            if s>0:
                Scale_layers = Scale_layers + [get_scale(X,s-1,d-1,Up=True)]

            # print('before block (s;d) = ({0},{1})'.format(s,d))
            return self.Block_table[s*self.Ndepth+d](Depth_layers+Scale_layers,Up,Down)

        #Generating latent code at each scale
        Xscales = [None]*self.Nscale
        for s in range(self.Nscale):
            Xscales[s] = self.FromImg[s](GenImg[s])

        Probas = [None]*self.Nscale
        Features = [None]*self.Nscale
        #Pass throug the total network
        for s in range(self.Nscale):
            Features[s],Probas[s] = self.ToProba[s](get_scale(Xscales,s,self.Ndepth-1))

        # IntermStates = Xscales
        # #Split the Multiscale Blocks in indepandant blocks in order to reduce
        # # the number of parameters
        # for n in range(self.N_MSBlocks):
        #     IntermStates_new = [None]*self.Nscale
        #     #Pass throug the total network
        #     for s in range(self.Nscale):
        #         IntermStates_new[s] = get_scale(IntermStates,s,self.Ndepth-1)
        #
        #     IntermStates = IntermStates_new
        #
        # Probas = [None]*self.Nscale
        # Features = [None]*self.Nscale
        # for s in range(self.Nscale):
        #     Features[s],Probas[s] = self.ToProba[s](IntermStates[s])

        return Features,Probas


class FromImg(nn.Module):
    def __init__(self,NFilter,SlopLRelu):
        super(FromImg, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=3,out_channels=NFilter,kernel_size=1,padding=0)
        self.LRelu = nn.LeakyReLU(SlopLRelu)
        self.Conv2 = nn.Conv2d(in_channels=NFilter,out_channels=NFilter,kernel_size=3,padding=1)
        self.ConvBN = nn.BatchNorm2d(NFilter)

    def forward(self,X):

        X = self.Conv1(X)
        X = self.LRelu(X)

        X = self.ConvBN(X)
        X = self.LRelu(X)
        X = self.Conv2(X)

        return X

class ToProba(nn.Module):

    def __init__(self,NFilter,scale,Nfeature,SlopLRelu):
        super(ToProba, self).__init__()

        self.Conv = nn.Conv2d(in_channels=NFilter,out_channels=NFilter,kernel_size=3,padding=1)
        self.LRelu = nn.LeakyReLU(SlopLRelu)

        self.Linear_Feat = nn.Linear(NFilter*scale**2,Nfeature)
        self.Linear_Ptrue = nn.Linear(Nfeature,1)

        self.Sig = nn.Sigmoid()

    def forward(self,X):

        X = self.Conv(X)
        X = self.LRelu(X)

        X = X.view(X.size(0), -1)

        Features = self.Linear_Feat(X)
        Ptrue = self.Sig(self.Linear_Ptrue(self.LRelu(Features))) #Probability to have a real image

        return Features,Ptrue
