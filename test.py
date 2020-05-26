import os,sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from modules import model
from modules.fast_context_model import Context4
from utilis import torch_msssim

# Load model
image_comp = model.Image_Coder_Context(M=192,N2=192)
pretrained_model = torch.load('ae.pkl',map_location='cpu')
image_comp.load_state_dict(pretrained_model.module.state_dict())
scaler = torch.load('scaler.pkl',map_location='cpu')

msssim_func = torch_msssim.MS_SSIM(max_val=1).cuda()

def main(im_dir, rec_dir, GPU=False):
    print('====> Encoding Image:', im_dir)
    
    img = Image.open(im_dir)
    img = np.array(img)/255.0
    H, W, _ = img.shape
    
    C = 3

    H_PAD = int(64.0 * np.ceil(H / 64.0))
    W_PAD = int(64.0 * np.ceil(W / 64.0))
    im = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    im[:H, :W, :] = img[:,:,:3]
    im = torch.FloatTensor(im)

    if GPU:
        image_comp.cuda()
        scaler.cuda()
        im = im.cuda()

    im = im.permute(2, 0, 1).contiguous()
    im = im.view(1, C, H_PAD, W_PAD)
    print("====> Image Info: Origin Size %dx%d, Padded Size: %dx%d"%(H,W,H_PAD,W_PAD))
    
    with torch.no_grad():
        y_main = image_comp.encoder.main_enc(im)
        y_main_q = scaler.decompress(torch.round(scaler.compress(y_main)))
        
        y_hyper = image_comp.encoder.hyper_enc(y_main)
        output = image_comp.decoder(y_main_q)
        y_hyper_q = torch.round(y_hyper)
        p_hyper = image_comp.factorized_entropy_func.likeli(y_hyper_q, quan_step = 1.0)
        
        p_main = image_comp.hyper_dec(y_hyper_q)
        p_main, _ = image_comp.context(y_main_q, p_main, quan_step = 1.0 / scaler.factor)

        bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * (H*W))
        bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * (H*W))
        bpp = bpp_hyper + bpp_main

        output_ = torch.clamp(output, min=0., max=1.0)
        out = output_.data[0].cpu().numpy()
        out = np.round(out * 255.0) 
        out = out.astype('uint8')
        output = out.transpose(1, 2, 0)   
        
        #ms-ssim
        mssim = msssim_func(im.cuda(),output_.cuda())
        
        #psnr
        mse =  torch.mean((im - torch.Tensor([out/255.0]).cuda()) * (im - torch.Tensor([out/255.0]).cuda()))
        psnr = 10. * np.log(1.0/mse.item())/ np.log(10.)
        img = Image.fromarray(output[:H, :W, :])
        img.save(rec_dir)

    return bpp.item(), mssim.item(), psnr

if __name__ == '__main__':
    # from glob import glob
    # bpps, msssims = 0., 0.
    # for i in glob("/kodak/*"):
    #     bpp, msssim, psnr = main(i, "test.png", GPU = True)
    #     bpps += bpp
    #     msssims += msssim

    # print(bpps/24., -10. * np.log10(1.-msssims/24.))
    
    bpp, msssim, psnr = main(sys.argv[1], sys.argv[2], GPU = True)
    print("bpp: %0.4f, PSNR: %0.4f, MS-SSIM (dB): %0.4f"%(bpp,psnr,msssim))
