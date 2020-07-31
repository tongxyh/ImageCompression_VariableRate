import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from utils import torch_msssim
from modules import model

# from utils import dali
# from nvidia.dali.plugin.pytorch import DALIGenericIterator

class SimpleDataset(Dataset):
    def __init__(self, input_path, img_size = 256):
        super(SimpleDataset, self).__init__()
        self.input_list = []
        self.label_list = []
        self.num = 0
        self.img_size = img_size

        for _ in range(30):
            for i in os.listdir(input_path):
                input_img = input_path + i
                self.input_list.append(input_img)
                self.num = self.num + 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = np.array(Image.open(self.input_list[idx]))
        input_np = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        input_tensor = torch.from_numpy(input_np)
        return input_tensor

class MyDataset(Dataset):
    def __init__(self, input_path, img_size = 256):
        super(MyDataset, self).__init__()
        self.input_list = []
        self.label_list = []
        self.num = 0
        self.img_size = img_size

        for i in os.listdir(input_path):
            input_img = input_path + i
            self.input_list.append(input_img)
            self.num = self.num + 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = np.array(Image.open(self.input_list[idx]))
        x = np.random.randint(0, img.shape[0] - self.img_size)
        y = np.random.randint(0, img.shape[1] - self.img_size)
        input_np = img[x:x + self.img_size, y:y + self.img_size, :].astype(np.float32).transpose(2, 0, 1) / 255.0
        input_tensor = torch.from_numpy(input_np)
        return input_tensor

def eval():
    pass

train_data = SimpleDataset(input_path='/datasets/img256x256/')
train_loader = DataLoader(train_data, batch_size=12, shuffle=True,num_workers=8)
# pipe = dali.SimplePipeline('../datasets', batch_size=12, num_threads = 2, device_id = 0)
# pipe.build()
# train_loader = DALIGenericIterator(pipe, ['data'], size=90306)

TRAINING = True
CONTEXT = True
M = 192
N2 = 192
image_comp = model.Image_Coder_Context(M=M,N2=N2).cuda()
image_comp = nn.DataParallel(image_comp,device_ids=[0,1])

METRIC = "MSSSIM"
print("====> using metric", METRIC)
SINGLE_MODEL = True
LOAD_EXIST,LOAD_SCALE = True, True
lamb = 2.
lr = 3e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if LOAD_EXIST:
    image_comp = torch.load('ae.pkl')

#params = list(image_comp.parameters()) + list(context.parameters())
if SINGLE_MODEL:
    print("====> traning single model scaler")
    if LOAD_EXIST:
        if LOAD_SCALE:
            scaler = torch.load('scaler.pkl')
            # scaler_hyper = torch.load('params/scaler_hyper.pkl')
        else:
            scaler = model.Scaler(channels = M).cuda()
            # scaler_hyper = model.Scaler(channels = N2).cuda()
        # params = list(scaler.parameters()) + list(scaler_hyper.parameters())
        optimizer = torch.optim.Adam(scaler.parameters(), lr=lr)
    else:
        raise Exception("Need to Load Pretrained Model!")
else:
    optimizer = torch.optim.Adam(image_comp.parameters(),lr=lr)

if METRIC == "MSSSIM":
    loss_func = torch_msssim.MS_SSIM(max_val=1).cuda()
elif METRIC == "PSNR":
    loss_func = nn.MSELoss()

for epoch in range(400):
    rec_loss, bpp = 0., 0.
    for step, batch_x in enumerate(train_loader):
        # batch_x = batch_x[0]['data']
        # batch_x = batch_x.type(dtype=torch.float32)
        # batch_x = torch.cast(batch_x,"float")/255.0
        # batch_x = batch_x/255.0
        batch_x = batch_x.cuda()
        num_pixels = batch_x.size()[0]*batch_x.size()[2]*batch_x.size()[3]

        # Training = True, CONTEXT = True
        if SINGLE_MODEL:
            with torch.no_grad():
                y_main, y_hyper = image_comp.module.encoder(batch_x.cuda())
            y_main_q = scaler.decompress(image_comp.module.add_noise(scaler.compress(y_main)))
            
            rec = image_comp.module.decoder(y_main_q)

            y_hyper_q, p_hyper = image_comp.module.factorized_entropy_func(y_hyper, TRAINING) #Training = True
            
            # TODO: scale here
            # y_hyper_q = scaler_hyper.decompress(image_comp.module.add_noise(scaler_hyper.compress(y_hyper)))
            # p_hyper = image_comp.module.factorized_entropy_func.likeli(y_hyper_q, quan_step = 1.0/scaler_hyper.factor ) #Training = True

            p_main = image_comp.module.hyper_dec(y_hyper_q)
            if CONTEXT:
                p_main, _ = image_comp.module.context(y_main_q, p_main, quan_step = 1.0 / scaler.factor)
            else:
                p_main = image_comp.module.gaussian_entropy_func(y_main_q, p_main, quan_step = 1.0 / scaler.factor)
 
        else:
            rec, y_main_q, y_hyper, p_main, p_hyper = image_comp(batch_x, TRAINING, CONTEXT)
        
        if METRIC == "MSSSIM":
            dloss = 1. - loss_func(rec, batch_x)
        elif METRIC == "PSNR":
            dloss = loss_func(rec, batch_x)

        train_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        train_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)

        loss = lamb * dloss + train_bpp_main + train_bpp_hyper

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if METRIC == "MSSSIM":
            rec_loss = rec_loss + (1. - dloss.item())
            d = 1. - dloss.item()
        elif METRIC == "PSNR":
            rec_loss = rec_loss + dloss.item()
            d = dloss.item()

        bpp = bpp+train_bpp_main.item()+train_bpp_hyper.item()

        print('epoch',epoch,'step:', step, '%s:'%(METRIC), d, 'main_bpp:',train_bpp_main.item(),
              'hyper_bpp:',train_bpp_hyper.item())

        cnt = 1000
        if (step+1) % cnt == 0:
            if SINGLE_MODEL:
                torch.save(scaler, 'scaler_%d_%d_%.8f_%.8f.pkl' % (epoch, step, rec_loss/cnt, bpp/cnt))
                torch.save(scaler_hyper, 'scaler_hyper_%d_%d_%.8f_%.8f.pkl' % (epoch, step, rec_loss/cnt, bpp/cnt))
            else:
                torch.save(image_comp, 'ae_%d_%d_%.8f_%.8f.pkl' % (epoch, step, rec_loss/cnt, bpp/cnt))
            rec_loss, bpp = 0., 0.