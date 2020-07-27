import torch
import torch.nn as nn
import numpy as np
from modules.factorized_entropy_model import Entropy_bottleneck
from modules.gaussian_entropy_model import Distribution_for_entropy2
from modules.basic_module import ResBlock, Non_local_Block
from modules.fast_context_model import Context4

class Trunk(nn.Module):
    
    # Non-Local Attention Module
    # Parameters:
    #       resblock_channels: input & output channels of resblock
    #       M1_block
    #       M2_block
    #       FLAG_NONLOCAL
    def __init__(self, resblock_channels, Trunk_blocks, Attention_blocks, FLAG_NONLOCAL):
        super(Trunk,self).__init__()
        self.N = int(resblock_channels)
        self.M1 = int(Trunk_blocks)
        self.M2 = int(Attention_blocks)
        self.FLAG = bool(FLAG_NONLOCAL)
        
        # main trunk
        self.trunk = nn.Sequential()
        for i in range(self.M1):
            self.trunk.add_module('res1'+str(i),ResBlock(self.N,self.N,3,1,1))
        
        # attention branch
        self.nlb = Non_local_Block(self.N, self.N // 2)
        self.attention = nn.Sequential()
        for i in range(self.M2):
            self.attention.add_module('res2'+str(i),ResBlock(self.N,self.N,3,1,1))
        self.attention.add_module('conv1',nn.Conv2d(self.N,self.N,1,1,0))

    def forward(self, x):
        if self.FLAG == False:
            attention = self.attention(x)
        else:
            attention = self.attention(self.nlb(x))
        return self.trunk(x) * torch.sigmoid(attention) + x

class Enc(nn.Module):
    def __init__(self,num_features,M1,M,N2):
        super(Enc,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)

        # main encoder
        self.conv1 = nn.Sequential(nn.Conv2d(self.n_features,self.M1,5,1,2),nn.ReLU())
        self.trunk1 = Trunk(self.M1,2,2,False)
        self.down1 = nn.Conv2d(self.M1,2*self.M1,5,2,2)
        self.trunk2 = Trunk(2*self.M1,4,4,False)
        self.down2 = nn.Conv2d(2 * self.M1, self.M, 5, 2, 2)
        self.trunk3 = Trunk(self.M, 4, 4, FLAG_NONLOCAL = True)
        self.down3 = nn.Conv2d(self.M, self.M, 5, 2, 2)
        self.trunk4 = Trunk(self.M, 4, 4,False)
        self.down4 = nn.Conv2d(self.M, self.M, 5, 2, 2)
        self.trunk5 = Trunk(self.M, 4, 4, FLAG_NONLOCAL = True)

        # hyper encoder
        self.trunk6 = Trunk(self.M,3,3,True)
        self.down6 = nn.Conv2d(self.M,self.M,5,2,2)
        self.trunk7 = Trunk(self.M,3,3,True)
        self.down7 = nn.Conv2d(self.M,self.M,5,2,2)
        self.conv2 = nn.Conv2d(self.M, self.N2, 3, 1, 1)
        self.trunk8 = Trunk(self.N2,3,3,True)
    
    def main_enc(self, x):
        x1 = self.conv1(x)
        x1 = self.down1(self.trunk1(x1))
        x2 = self.down2(self.trunk2(x1))
        x3 = self.down3(self.trunk3(x2))
        x4 = self.down4(self.trunk4(x3))
        x5 = self.trunk5(x4)
        return x5

    def hyper_enc(self, x):
        x6 = self.down6(self.trunk6(x))
        x7 = self.down7(self.trunk7(x6))
        x8 = self.trunk8(self.conv2(x7))
        return x8

    def forward(self, x):
        x5 = self.main_enc(x)
        x8 = self.hyper_enc(x5)
        return [x5,x8]

class Hyper_Dec(nn.Module):
    def __init__(self, N2,M):
        super(Hyper_Dec, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        # hyper decoder
        self.trunk8 = Trunk(self.N2, 3, 3, True)
        self.conv2 = nn.Conv2d(self.N2, self.M, 3, 1, 1)
        self.up7 = nn.ConvTranspose2d(self.M, self.M, 5, 2, 2, 1)
        self.trunk7 = Trunk(self.M, 3, 3, True)
        self.up6 = nn.ConvTranspose2d(self.M, self.M, 5, 2, 2, 1)
        self.trunk6 = Trunk(self.M, 3, 3, True)
        self.conv3 = nn.Conv2d(self.M,2*self.M,3,1,1)

    def forward(self,xq2):
        x7 = self.conv2(self.trunk8(xq2))
        x6 = self.trunk7(self.up6(x7))
        x5 = self.trunk6(self.up7(x6))
        x5 = self.conv3(x5)
        return x5

class Dec(nn.Module):
    def __init__(self,num_features,M1,M):
        super(Dec,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)

        # main decoder
        self.trunk5 = Trunk(self.M, 4, 4, FLAG_NONLOCAL = True)
        self.up4 = nn.ConvTranspose2d(self.M, self.M, 5, 2, 2,1)
        self.trunk4 = Trunk(self.M, 4, 4, False)
        self.up3 = nn.ConvTranspose2d(self.M, self.M, 5, 2, 2,1)
        self.trunk3 = Trunk(self.M, 4, 4, FLAG_NONLOCAL = True)
        self.up2 = nn.ConvTranspose2d(self.M, 2*self.M1, 5, 2, 2,1)
        self.trunk2 = Trunk(2 * self.M1, 4, 4, False)
        self.up1 = nn.ConvTranspose2d(2*self.M1, self.M1, 5, 2, 2,1)
        self.trunk1 = Trunk(self.M1, 2, 2, False)
        self.conv1 = nn.Conv2d(self.M1, self.n_features,  5, 1, 2)

    def forward(self,xq1):
        x5 = self.up4(self.trunk5(xq1))
        x4 = self.up3(self.trunk4(x5))
        x3 = self.up2(self.trunk3(x4))
        x2 = self.up1(self.trunk2(x3))
        x1 = self.trunk1(x2)
        x = self.conv1(x1)
        return x

class Scaler(nn.Module):
    def __init__(self, channels):
        super(Scaler,self).__init__()
        self.bias = nn.Parameter(torch.zeros([1,channels,1,1]))
        self.factor = nn.Parameter(torch.ones([1,channels,1,1]))

    def compress(self,x):
        return self.factor * (x - self.bias)

    def decompress(self,x):
        return self.bias + x / self.factor

class Image_coding(nn.Module):
    def __init__(self,M,N2,num_features=3,M1=32):
        super(Image_coding,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.encoder = Enc(num_features, self.M1, self.M, self.N2)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.decoder = Dec(num_features, self.M1,self.M)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self,x,if_training):
        y_main, y_hyper = self.encoder(x)
        y_hyper_q, p_hyper = self.factorized_entropy_func(y_hyper, if_training)
        gaussian_params = self.hyper_dec(y_q_hyper)
        if if_training:
            y_main_q = self.add_noise(y_main)
        else:
            y_main_q = torch.round(y_main)
        p_main = self.gaussin_entropy_func(y_main_q, gaussian_params)
        output = self.decoder(y_main_q)

        return output, p_main, p_hyper, y_main_q, gaussian_params

class Image_Coder_Context(nn.Module):
    def __init__(self,M,N2,num_features=3,M1=32):
        super(Image_Coder_Context,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.encoder = Enc(num_features, self.M1, self.M, self.N2)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.gaussian_entropy_func = Distribution_for_entropy()
        self.context = Context4(M)
        self.decoder = Dec(num_features, self.M1,self.M)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self,x, if_training, CONTEXT):
        y_main, y_hyper = self.encoder(x)

        if if_training:
            y_main_q = self.add_noise(y_main)
        else:
            y_main_q = torch.round(y_main)
        
        output = self.decoder(y_main_q)

        y_hyper_q, p_hyper = self.factorized_entropy_func(y_hyper, if_training) #Training = True
        p_main = self.hyper_dec(y_hyper_q)
        if CONTEXT:
            p_main, _ = self.context(y_main_q, p_main)
            #p_main = self.gaussian_entropy_context(y_main_q, p_main)
        else:
            p_main = self.gaussian_entropy_func(y_main_q, p_main)

        # y_hyper_q, p_hyper = self.factorized_entropy_func(y_hyper, if_training)
        # gaussian_params = self.hyper_dec(y_q_hyper)
        # p_main = self.gaussin_entropy_func(y_main_q, gaussian_params)
        return output, y_main_q, y_hyper, p_main, p_hyper