# coding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim

from torch.autograd import Variable
from torch.nn.init import kaiming_normal, calculate_gain
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from utils import *
import time
from models import MSGenerator, MSDiscriminator
from losses import *
# from logger import Logger
from torch.utils.data import Dataset,random_split,DataLoader
# from Debug_model import *

from datetime import datetime

from tqdm import tqdm

# import torch.multiprocessing as mp
# mp.set_start_method('spawn') # Avoid same seed for multiprocessing


class MSGAN:
    '''
        Class that include all the parameters of the optimisation and
        save the model after each epoch
    '''

    def __init__(self,data_folder,Nepochs=1000,SlopLRelu = 0.2,use_cuda=True):
        '''
            SamplesFile: Location of the hdf5 file with all the samples
            Nepochs: Number of epochs
            balance: Balance in the final loss for the generator
            WParam: Parameter that attirbute more weigth to the diagonal because
                    of the increase in difficulty
            startcounter:   Start the counter of the number of iteration at startcounter.
                            Allow to resume the learning of a model

        '''
        self.use_cuda = use_cuda

        self.latent_dim = 10

        self.Nepochs = Nepochs
        self.StartEpochs = 0 #Different from zeros if the optimisation is resuming

        self.SlopLRelu = SlopLRelu

        self.NF = 32
        self.Ndepth = 5
        self.Ndepth_max = 5
        self.scales = [4,8,16,32]
        self.Nfeatures = [16,32,64,128]
        # self.len_ohe = 10

        self.depths = [i for i in range(self.Ndepth)]
        self.Nscales = len(self.scales)

        self.G = MSGenerator(NF=self.NF,
                             scales=self.scales,
                             depths=self.depths,
                             Ndepth_max=self.Ndepth_max,
                             SlopLRelu=self.SlopLRelu,
                             latent_dim=self.latent_dim)
                             # len_ohe=self.len_ohe)

        self.D = MSDiscriminator(NF=self.NF,
                                scales=self.scales,
                                depths=self.depths,
                                Ndepth_max=self.Ndepth_max,
                                SlopLRelu=self.SlopLRelu,
                                Nfeatures=self.Nfeatures)
                                # len_ohe=self.len_ohe)

        self.optim_G = optim.Adam(self.G.parameters(),lr=0.0002,betas=(0.5,0.999))
        self.optim_D = optim.Adam(self.D.parameters(),lr=0.0002,betas=(0.5,0.999))

        self.batchsize = 8

        self.compute_adv_loss = WGANLoss()#.cuda()

        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.compute_adv_loss = self.compute_adv_loss.cuda()

        self.reg_param = 10. #regularization parameter

        #Init Logger for the tensorboard
        # self.logger = Logger('./logs'  + "/") #Logger for the scalar & histograms
        self.writer = SummaryWriter('./logs/') #Writter for image saving

        self.g_loss_record = {key:[] for key in ['train','val']}
        self.g_adv_loss_record = {key:[] for key in ['train','val']}
        self.d_loss_record = {key:[] for key in ['train','val']}
        self.d_adv_loss_fake_record = {key:[] for key in ['train','val']}
        self.d_adv_loss_real_record = {key:[] for key in ['train','val']}

        #Generator/Discriminator loss mean as indicator for the optimal model
        self.best_g_loss = np.power(10,20) #np.inf
        self.best_d_loss = np.power(10,20) #np.inf


        self.get_training_images = True
        self.ImagesDir = './TrainingImages/'
        if not os.path.exists(self.ImagesDir): os.makedirs(self.ImagesDir)
        self.ModelDir = './TrainingModels/'
        if not os.path.exists(self.ModelDir): os.makedirs(self.ModelDir)

        #Define dataloader
        train_dataset,test_dataset = load_CIFAR10_datasets(data_folder,
                                                           self.latent_dim,
                                                           self.Nscales)
        self.train_loader = DataLoader(train_dataset,
                                        batch_size=self.batchsize,
                                        shuffle=True,
                                        num_workers=30,
                                        pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
                                        batch_size=self.batchsize,
                                        shuffle=True,
                                        num_workers=30,
                                        pin_memory=True)

        self.train_len = len(train_dataset)
        self.test_len = len(test_dataset)

        #Load label names
        self.label_names = load_CIFAR10_label_names(data_folder)

        #Fixed latent code to generate always the same image
        # to follow the evolution of the training
        self.fixed_LC = [np.random.randn(1,self.latent_dim).astype(np.float32)
                        for _ in range(self.Nscales)]

        print("Initialize the networks weigths...")
        self.G.apply(self.he_init)
        self.D.apply(self.he_init)


    def train(self):

        for self.epoch in tqdm(range(self.StartEpochs,self.Nepochs)):

            self.G.train()
            self.D.train()
            self.phase = 'train'


            self.clear_loss_records()
            total_iter = np.ceil(self.train_len/self.batchsize)
            for self.counter,(self.X,_,self.latent_codes) in enumerate(tqdm(self.train_loader,total=total_iter,desc='train')):

                self.X = Variable(self.X,requires_grad=True)#.cuda()
                self.latent_codes = [Variable(latent_code) for latent_code in self.latent_codes]

                if self.use_cuda:
                    self.X = self.X.cuda()
                    self.latent_codes = [latent_code.cuda() for latent_code in self.latent_codes]

                #Pooling Real image to fit generator Output
                factors = [int(self.scales[-1]/scale) for scale in self.scales]
                self.Img_real = [nn.AvgPool2d(factor,stride=factor,padding=0)(self.X) for factor in factors]


                # ===update D===
                self.optim_D.zero_grad()
                self.forward_D()
                self.backward_D()

                # ===update G===
                self.forward_D()
                self.optim_G.zero_grad()
                self.forward_G()
                self.backward_G()

                # print 'record loss'
                self.record_loss()
                self.StartEpochs = self.epoch

                # if self.counter > 1000:
                #     break

            # ===validation===
            self.validate()
            # ===tensorboard visualization===
            self.tensorboard()

            # ===save model===
            self.save()


    def forward_D(self):

        self.Img_fake = self.G(self.latent_codes)
        _,self.d_real = self.D(self.Img_real)
        _,self.d_fake = self.D([Img.detach() for Img in self.Img_fake])
        #detach means that the netork is using HRfake but block the optimization
        #to the network that generated it, ie clone the variable as it's a new one

    def forward_G(self):
        _,self.d_fake = self.D(self.Img_fake)


    def backward_G(self):
        self.g_loss = self.compute_G_loss()
        self.g_loss.backward()
        self.optim_G.step()


    def backward_D(self):
        self.d_loss = self.compute_D_loss()

        #retain_graph=False because the loss function
        # for the gradient and the discriminator are not
        # the same and therefore the gradients are differents

        # self.d_loss.backward(retain_graph=False)
        self.d_loss.backward(retain_graph=True)
        self.d_real_reg = sum([self.reg_param * compute_grad2(d_real_i, Img_real_i).mean() for d_real_i,Img_real_i in zip(self.d_real,self.Img_real)])
        self.d_real_reg.backward()
        # R1_reg(dloss_real,d_real,x_real)
        self.optim_D.step()


    def compute_G_loss(self):
        #Make the discr find True when fake
        Nlayers = len(self.d_fake)
        # learning_factors = [numpy2var((Nlayers-(i+1))*((self.epoch+1)/self.Nmax_epoch),use_cuda=True) for i in range((Nlayers))]
        # self.g_adv_loss = [self.compute_adv_loss(self.d_fake[i], True)*learning_factors[i] for i in range(len(self.d_fake))]
        self.g_adv_loss = [self.compute_adv_loss(d_fake_i, True) for d_fake_i in self.d_fake]

        #Concatenate all losses
        self.g_adv_loss = sum(self.g_adv_loss)

        return self.g_adv_loss

    def compute_D_loss(self):

        # Nlayers = len(self.d_real)
        # self.d_adv_loss = []
        # for i in range(len(self.d_real)):
        #     self.d_adv_loss_real = self.compute_adv_loss(self.d_real[i], True)
        #     self.d_adv_loss_fake = self.compute_adv_loss(self.d_fake[i], False)
        #     # learning_factor = numpy2var((Nlayers-(i+1))*((self.epoch+1)/self.Nmax_epoch),use_cuda=True)
        #     # self.d_adv_loss.append((self.d_adv_loss_real + self.d_adv_loss_fake)*learning_factor)
        #     self.d_adv_loss.append((self.d_adv_loss_real + self.d_adv_loss_fake))
        #
        # #Concatenate all losses
        # self.d_adv_loss = sum(self.d_adv_loss)
        self.d_adv_loss_real = sum([self.compute_adv_loss(d_real_i, True) for d_real_i in self.d_real])
        self.d_adv_loss_fake = sum([self.compute_adv_loss(d_fake_i, False) for d_fake_i in self.d_fake])

        #Regularization on the gradient of real samples
        # self.d_adv_loss_real = self.d_adv_loss_real + self.d_real_reg


        self.d_adv_loss = (self.d_adv_loss_real + self.d_adv_loss_fake)/(2.*len(self.d_real))

        return self.d_adv_loss

    def record_loss(self):
        p = self.phase
        self.g_loss_record[p].append(var2numpy(self.g_loss.mean(),use_cuda=self.use_cuda))
        self.d_loss_record[p].append(var2numpy(self.d_loss.mean(),use_cuda=self.use_cuda))
        self.d_adv_loss_fake_record[p].append(var2numpy(self.d_adv_loss_fake.mean(),use_cuda=self.use_cuda))
        self.d_adv_loss_real_record[p].append(var2numpy(self.d_adv_loss_real.mean(),use_cuda=self.use_cuda))

    def clear_loss_records(self):
        for p in ['train','val']:
            self.g_loss_record[p] = []
            self.d_loss_record[p] = []
            self.d_adv_loss_fake_record[p] = []
            self.d_adv_loss_real_record[p] = []


    def validate(self):

        self.G.eval()
        self.D.eval()
        self.phase = 'val'
        total_iter = np.ceil(self.test_len/self.batchsize)

        for self.counter,(self.X,_,self.latent_codes) in enumerate(tqdm(self.test_loader,total=total_iter,desc='validation')):

            #Generate latent code
            self.X = Variable(self.X,requires_grad=True)#.cuda()
            # self.X = Variable(self.X)#.cuda()
            self.latent_codes = [Variable(latent_code) for latent_code in self.latent_codes]

            if self.use_cuda:
                self.X = self.X.cuda()
                self.latent_codes = [latent_code.cuda() for latent_code in self.latent_codes]


            with torch.no_grad():
                #Pooling Real image to fit generator Output
                # factors = [int(self.ImgSizes[-1]/imgsize) for imgsize in self.ImgSizes]
                factors = [int(self.scales[-1]/scale) for scale in self.scales]
                self.Img_real = [nn.AvgPool2d(factor,stride=factor,padding=0)(self.X) for factor in factors]

                self.forward_D()
                self.forward_G()
                self.g_loss = self.compute_G_loss()
                self.d_loss = self.compute_D_loss()
                self.record_loss()

            # if self.counter > 1000:
            #     break

    def predict(self,img,labels,batchsize=1):

        dataset = dataset_h5(img,labels,self.latent_dim,self.Nscales)
        data_loader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=False,
                                num_workers=30,
                                pin_memory=True)

        self.D.eval()
        total_iter = np.ceil(img.shape[0]/batchsize)

        features = [ [] for _ in self.scales]

        for counter,(X,_,_) in enumerate(tqdm(data_loader,total=total_iter,desc='prediction')):

            X = Variable(X)
            if self.use_cuda:
                X = X.cuda()

            with torch.no_grad():
                #Pooling Real image to fit generator Output
                factors = [int(self.scales[-1]/scale) for scale in self.scales]
                Img_real = [nn.AvgPool2d(factor,stride=factor,padding=0)(X) for factor in factors]
                for i,feat in enumerate(self.D(Img_real)[0]):
                    if self.use_cuda:
                        feat = feat.cpu()
                    features[i].append(feat.data.numpy()[0])


        features = [np.vstack(feat) for feat in features]

        return features


    def generate(self,Nimages=1,latent_codes=None):

        self.G.eval()

        get_latent=True if latent_codes==None else False

        gen_image = [[] for i in range(Nimages)]
        for i in range(Nimages):
            #Generate latent code
            seed = np.random.seed(i+datetime.now().second + datetime.now().microsecond)

            if get_latent:
                latent_codes = [np.random.randn(1,self.latent_dim).astype(np.float32)
                               for _ in range(self.Nscales)]

            latent_codes = [Variable(torch.from_numpy(latent_code)) for latent_code in latent_codes]
            if self.use_cuda:
                latent_codes = [latent_code.cuda() for latent_code in latent_codes]

            with torch.no_grad():
                #Pooling Real image to fit generator Output
                # factors = [int(self.ImgSizes[-1]/imgsize) for imgsize in self.ImgSizes]

                for X in self.G(latent_codes):
                    if self.use_cuda:
                        X = X.cpu()
                        gen_image[i].append(X.data.numpy()[0].transpose((1,2,0)))

        return gen_image

    def save(self):
        file_name = os.path.join(self.ModelDir, 'Epoch%d' % (self.epoch))
        g_file = file_name + '-G.pth'
        d_file = file_name + '-D.pth'

        g_loss_mean = np.array(self.g_loss_record['val']).mean()
        d_loss_mean = np.array(self.d_loss_record['val']).mean()

        # if g_loss_mean<self.best_g_loss:
        if True:
            state = {'state_dict': self.G.state_dict(),
                     'optimizer': self.optim_G.state_dict(),
                     'epoch': self.epoch,
                    }
            torch.save(state, g_file)
            self.best_g_loss = g_loss_mean

        # if d_loss_mean<self.best_d_loss:
        if True:
            state = {'state_dict': self.D.state_dict(),
                     'optimizer': self.optim_D.state_dict(),
                     'epoch': self.epoch,
                    }
            torch.save(state, d_file)
            self.best_d_loss = d_loss_mean

    def load(self,Gpath,Dpath):
        state_g = torch.load(Gpath)
        self.G.load_state_dict(state_g['state_dict'])
        self.optim_G.load_state_dict(state_g['optimizer'])


        state_d = torch.load(Dpath)
        self.D.load_state_dict(state_d['state_dict'])
        self.optim_D.load_state_dict(state_d['optimizer'])

        #Reset the best loss for the generator and discriminator
        self.best_g_loss = np.power(10,20) #np.inf
        self.best_d_loss = np.power(10,20) #np.inf

    def tensorboard(self):
        # ===Add scalar losses===
        for p in ['train','val']:
            prefix = p+'/'
            info = {prefix + 'G_loss': np.array(self.g_loss_record[p]).mean(),
                    prefix + 'D_loss': np.array(self.d_loss_record[p]).mean(),
                    prefix + 'D_adv_loss_fake': np.array(self.d_adv_loss_fake_record[p]).mean(),
                    prefix + 'D_adv_loss_real': np.array(self.d_adv_loss_real_record[p]).mean()}

            # self.writer.add_scalars(p, info, self.epoch)
            for tag, value in info.items():
                self.writer.add_scalars(tag, {tag:value}, self.epoch)
                # self.logger.scalar_summary(tag, value, self.epoch)

        # ===Add gradien histogram===
        for tag, value in self.G.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('G/' + prefix +tag, var2numpy(value), self.epoch)
            # self.logger.histo_summary('G/' + prefix +tag, var2numpy(value), self.epoch)
            if value.grad is not None:
                self.writer.add_histogram('G/' + prefix +tag + '/grad', var2numpy(value.grad), self.epoch)
                # self.logger.histo_summary('G/' + prefix +tag + '/grad', var2numpy(value.grad), self.epoch)

        for tag, value in self.D.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('D/' + prefix + tag, var2numpy(value), self.epoch)
            # self.logger.histo_summary('D/' + prefix + tag, var2numpy(value), self.epoch)
            if value.grad is not None:
                self.writer.add_histogram('D/' + prefix + tag + '/grad',var2numpy(value.grad), self.epoch)
                # self.logger.histo_summary('D/' + prefix + tag + '/grad',var2numpy(value.grad), self.epoch)

        #===generate sample images===
        if self.get_training_images == True:
            #
            # K = np.random.randint(self.batchsize)
            #
            # f,ax = plt.subplots(1,len(self.Img_fake),figsize=(int(5*len(self.Img_fake)),5))
            # for i in range(len(self.Img_fake)):
            #     if self.use_cuda:
            #         img = self.Img_fake[i].cpu()
            #     else:
            #         img = self.Img_fake[i]
            #
            #     img = (img.data.numpy()[K]).transpose((1,2,0))
            #

            gen_img = self.generate(Nimages=1,latent_codes=self.fixed_LC)[0]
            f,ax = plt.subplots(1,len(gen_img),figsize=(5*len(gen_img),5))
            for i,img in enumerate(gen_img):
                ax[i].imshow(img)

            plt.savefig(os.path.join(self.ImagesDir,'Img-Epoch%d.png' % (self.epoch)),format='png')


    def he_init(self,layer, nonlinearity='conv2d'):

        classname = layer.__class__.__name__

        # Check if the leayer is a convolution.
        # If True, apply Kaiming normalization
        if classname.find('Conv') != -1:
            nonlinearity = nonlinearity.lower()
            if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
                if not hasattr(layer, 'gain') or layer.gain is None:
                    gain = 0  # default
                else:
                    gain = layer.gain
            elif nonlinearity == 'leaky_relu':
                # assert param is not None, 'Negative_slope(param) should be given.'
                gain = calculate_gain(nonlinearity, self.SlopLRelu)
            else:
                gain = calculate_gain(nonlinearity)
                kaiming_normal(layer.weight, a=gain)


    def copy(self,model):
        '''
            Allow to get paramters from another pretrained model
        '''
        for key in model.__dict__.keys():
                self.__dict__[key] = model.__dict__[key]
