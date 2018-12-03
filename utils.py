import numpy as np
import pickle
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset
import datetime

from sklearn.preprocessing import StandardScaler,OneHotEncoder

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10(folder):

    img = []
    labels = []

    for i in range(1,6):
        data = unpickle(folder+'/data_batch_' + str(i))
        img.append(data[b'data'].reshape(data[b'data'].shape[0],3,32,32))
        labels.append(data[b'labels'])


    img = np.concatenate(img,axis=0).astype(np.float32)
    labels = np.concatenate(labels,axis=0).astype(np.float32)

    len_train = int(img.shape[0]*0.8)
    len_test = img.shape[0] - len_train
    train_img = img[:len_train]
    train_labels = labels[:len_train]
    test_img = img[len_train:]
    test_labels = labels[len_train:]

    return train_img,test_img,train_labels,test_labels

def load_CIFAR10_datasets(folder,latent_dim,Nscales):
    '''
        Load CIFAR 10 and return train and test datasets
    '''
    train_img,test_img,train_labels,test_labels = load_CIFAR10(folder)

    #train_dataset, test_dataset
    return dataset_h5(train_img, train_labels,latent_dim,Nscales),dataset_h5(test_img, test_labels,latent_dim,Nscales)



class dataset_h5(Dataset):
    def __init__(self,X,labels,latent_dim,Nscales):
        super(dataset_h5, self).__init__()

        '''
            Create the dataset and provide a generator for it
            WSPool: Size of the images patches after pooling
                  (Needs to be a multiple of the original windows size)
        '''

        self.Nscales = Nscales

        self.X = X/255. #From [0,255] to [0,1]
        self.X = rgb2gray(self.X)

        # for i in range(X.shape[1]):
        #     self.X[:,i,:,:] = X[:,i,:,:]-X[:,i,:,:].mean()/X[:,i,:,:].std()

        self.labels = labels
        self.latent_dim = latent_dim

        idx = np.where(labels==0)[0]
        self.X = self.X[idx]
        self.labels = self.labels[idx]

        # L = int(self.labels.max()+1)
        # self.OHE = OneHotEncoder(sparse=False,n_values=L).fit(self.labels.reshape(-1, 1))
        # self.X = self.X[:2000]
        # self.labels = self.labels[:2000]

    def __getitem__(self, index):
        #From tensor to numpy index
        #index = index.numpy()
        np.random.seed(datetime.datetime.now().second + index)

        x = self.X[index].astype('float32')
        l = self.labels[index].astype('float32')
        #
        # ohe = self.OHE.transform(l)
        # latent_codes = [np.concatenate([np.random.randn(1,self.latent_dim),ohe[None,:]],axis=1).astype(np.float32)
        #                for _ in range(self.Nscales)]
        latent_codes = [np.random.randn(1,self.latent_dim).astype(np.float32)
                       for _ in range(self.Nscales)]

        return x,l,latent_codes

    def __len__(self):
        return len(self.labels)


def load_CIFAR10_label_names(folder):
    return unpickle(folder+'batches.meta')[b'label_names']


def var2numpy(var,use_cuda=True):
    if use_cuda:
        return var.cpu().data.numpy()
    return var.data.numpy()

def numpy2var(nmpy,use_cuda=True):
    if type(nmpy) is not np.ndarray or type(nmpy) is not np.array:
        nmpy = np.array([nmpy],dtype=np.float32)
    var = Variable(torch.from_numpy(nmpy))
    if use_cuda:
        return var.cuda()
    return var

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def data_parallel(module, input, device_ids, output_device=None):
    '''
        Allow to launch the model over multiple GPUs in parallel
    '''
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


# def plot_CIFAR(tr_x,tr_y,ind):
#     '''
#         Display CIFAR image indexed by ind in tr_x labeled by tr_y
#     '''
#     arr = tr_x[ind].flatten()
#     sc_dpi = 157.35
#     R = arr[0:1024].reshape(32,32)/255.0
#     G = arr[1024:2048].reshape(32,32)/255.0
#     B = arr[2048:].reshape(32,32)/255.0
#
#     img = np.dstack((R,G,B))
#     title = re.sub('[!@#$b]', '', str(labels[tr_y[ind]]))
#     fig = plt.figure(figsize=(3,3))
#     ax = fig.add_subplot(111)
#     ax.imshow(img,interpolation='bicubic')
#     ax.set_title('Category = '+ title,fontsize =15)
