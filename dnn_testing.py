import torch
import torch.optim as optim
import random
import math
import numpy
from ChEst import ChEst
from ChEstDataset import ChEstDataset

#ADD JOSHUA'S AND THIS TO GITHUB REPO
#WORK ON README.MD

SNR = 20
sym_blk = 400
sr = 5000
num_files = 500
num_blk = math.floor(sr/sym_blk)
cir_length = 200

# Look into MATLAB DNN package to import Pytorch model
dev = ""
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
print(dev, "used")
net = torch.load("model")
# testset = ChEstDataset(r"Dataset", SNR, True)
testset = ChEstDataset(r"Dataset", SNR, False)
# files_ind = list(range(0,len(testset)))
files_ind = [0]
num_bits = 0
with torch.no_grad():
    for i, data in enumerate(files_ind,0):
        sample_indices = list(range(0,num_blk))
        est_channel = torch.zeros((num_blk*sym_blk,2*cir_length)).to(dev)
        act_channel = torch.zeros((num_blk*sym_blk,2*cir_length)).to(dev)
        sample = testset[i]
        for idx in iter(sample_indices):
            usb, r, h_ls, h = torch.tensor(sample['usb'][idx*sym_blk:(idx+1)*sym_blk]).to(dev), \
                sample['r'][idx*sym_blk:(idx+1)*sym_blk], \
                sample['h_ls'][idx], \
                sample['h'][idx*sym_blk:(idx+1)*sym_blk]
            usb = (usb+1)/2
            r = torch.tensor(numpy.concatenate((r.real,r.imag),axis=1)).to(dev)
            r = (r-torch.min(r))/(torch.max(r)-torch.min(r))
            h_ls = torch.tensor(numpy.concatenate((h_ls.real,h_ls.imag),axis=0)).repeat(sym_blk,1)
            h_ls = (h_ls-torch.min(h_ls))/(torch.max(h_ls)-torch.min(h_ls))
            blk_input = torch.cat((usb,r,h_ls),axis=1)
            est_channel = net(blk_input)*(torch.max(h_ls)-torch.min(h_ls))+torch.min(h_ls)
            act_channel[idx*sym_blk:(idx+1)*sym_blk,:] = h
        numpy.savetxt("h.csv", act_channel.to("cpu").numpy()/10, delimiter=",")
        numpy.savetxt("h_hat.csv", est_channel.to("cpu").numpy()/10, delimiter=",")
        print("File ", i+1, " Done!")
        print(num_bits)
        break
BER = math.log10(num_bits/(75*sym_blk*num_blk))
print(BER)
