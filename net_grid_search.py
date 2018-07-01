from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os

from models import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
import torch.utils.data as data
from PIL import Image
import numpy as np

class CIFAR10(data.Dataset):

    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo,encoding='latin1')
            x = dict['data']
            y = dict['labels']
        return x,y
    
    def loadData(self,train=True,noise=False):        
        if train==True:
            x_train=[]
            y_train=[]
            for i in range(5):           
                x,y=self.unpickle("../cifar-10-batches-py/data_batch_"+str(i+1))
                x_train.append(x)
                y_train += y
            x_train=np.concatenate(x_train)
            return x_train,y_train
        elif noise==False:
            x_test=[]
            y_test=[]
            x,y=self.unpickle("../cifar-10-batches-py/test_batch")
            x_test=x
            y_test=y
            return x_test,y_test
        else:
            import scipy.io
            noise = scipy.io.loadmat('X_noise.mat')
            x_noise=noise['X_noise']
            x_noise=x_noise.astype(np.uint8)
            y_noise=[]
            _,y=self.unpickle("../cifar-10-batches-py/test_batch")
            y_noise=y
            return x_noise,y_noise
           
    def __init__(self,train=True,transform=None,noise=False):
        
        self.train=train
        self.transform=transform
        ########################################################################
        if self.train:
            self.train_data,self.train_labels = self.loadData(train=True)

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            self.test_data,self.test_labels=self.loadData(train=False,noise=noise)
            
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        ########################################################################

    def __getitem__(self,index):
        
        if self.train:
            img,target=self.train_data[index],self.train_labels[index]
        else:
            img,target=self.test_data[index],self.test_labels[index]
            
        img=Image.fromarray(img)
        
        if self.transform is not None:
            img=self.transform(img)
        
        return img,target
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    # Load checkpoint.
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
noiseset=CIFAR10(train=False,transform=transform_test,noise=True)
noiseloader = torch.utils.data.DataLoader(noiseset, batch_size=100, shuffle=False, num_workers=2)
        
device = "cuda" if torch.cuda.is_available() else "cpu"
net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/ckpt.t7')
net.load_state_dict(checkpoint['net'])

net.eval()

net1=nn.Sequential(*list(net.module.features.children())[0:1])
net2=nn.Sequential(*list(net.module.features.children())[1:4])
net3=nn.Sequential(*list(net.module.features.children())[4:8])
net4=nn.Sequential(*list(net.module.features.children())[8:11])
net5=nn.Sequential(*list(net.module.features.children())[11:15])
net6=nn.Sequential(*list(net.module.features.children())[15:18])
net7=nn.Sequential(*list(net.module.features.children())[18:21])
net8=nn.Sequential(*list(net.module.features.children())[21:25])
net9=nn.Sequential(*list(net.module.features.children())[25:28])
net10=nn.Sequential(*list(net.module.features.children())[28:31])
net11=nn.Sequential(*list(net.module.features.children())[31:35])
net12=nn.Sequential(*list(net.module.features.children())[35:38])
net13=nn.Sequential(*list(net.module.features.children())[38:41])

net14=nn.Sequential(*list(net.module.features.children())[41:])
net15=net.module.classifier 

netList=[net1,net2,net3,net4,net5,net6,net7,net8,net9,net10,net11,net12,net13]
sizeList=[32,32,16,16,8,8,8,4,4,4,2,2,2]
##L=1~13
#31,31,15,15,7,7,7,3,3,3
#
#30,30,15,15,7,7,7,3,3,3
#
#29,29,14,14,7,7,7,3,3,3

class zzNet(nn.Module):

    def __init__(self,r,L):
        super(zzNet, self).__init__()
        self.r=r
        self.L=L

    def forward(self,x):
        x=torch.nn.Upsample(size=(self.r, self.r), mode='bilinear')(x)  
        net_pre=nn.Sequential(*netList[0:self.L])
        out=net_pre(x)
        r2=sizeList[self.L-1]
        out=torch.nn.Upsample(size=(r2,r2),mode='bilinear')(out)
        net_pro=nn.Sequential(*netList[self.L:])
        out=net_pro(out)
        out=net14(out)
        out = out.view(out.size(0), -1)
        out = net15(out)
        return out

def aaa(rsize,layer):
    zz=zzNet(rsize,layer)#16,6
    zz.to(device)
    zz.eval()
    correct = 0
    total = 0
   
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(noiseloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = zz(inputs)
    
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    #        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#            print("batch",batch_idx,"correct",correct/total)

    print(correct/total)
    return correct/total

#32~3 rsize
lenList=[2,4,7,10,13]

acc_mat=np.zeros((32,13))
for i in range(3-1,32):
    rsize=i+1
    fsize=rsize
    lenInd=0
    while fsize>1:
        fsize=fsize//2
        lenInd+=1
   
    layerLen=lenList[lenInd-1]
    
    for j in range(1-1,layerLen):
        layer=j+1
        acc=aaa(rsize,layer)
        acc_mat[i,j]=acc
        print(i," ",j)
        
##########################
np.save("accMat.npy", acc_mat)
acc_load = np.load("accMat.npy")

import matplotlib.pyplot as plt
plt.imshow((acc_mat-np.max(acc_mat))/(np.max(acc_mat)-np.min(acc_mat)))#,cmap='gray')

