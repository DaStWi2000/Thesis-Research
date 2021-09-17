import torch
import torch.nn as nn
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
net = ChEst(cir_length, sym_blk).to(torch.double).to(dev)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters())
optimizer.zero_grad()
trainset = ChEstDataset(r"Dataset", SNR, False)
# files_ind = list(range(0,len(trainset)))
files_ind = [0]
random.shuffle(files_ind)
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(files_ind,0):
        sample_indices = list(range(0,num_blk))
        est_channel = torch.zeros((2*cir_length,2*cir_length)).to(dev)
        act_channel = torch.zeros((2*cir_length,2*cir_length)).to(dev)
        optimizer.zero_grad()
        sample = trainset[i]
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
            h = torch.tensor(numpy.concatenate((h.real,h.imag),axis=1)).to(dev)
            h = (h-torch.min(h))/(torch.max(h)-torch.min(h))
            blk_input = torch.cat((usb,r,h_ls),axis=1)
            est_channel = net(blk_input)
            act_channel = h
            loss = criterion(est_channel, act_channel)
            if loss.item() != loss.item():
                print("ERROR", epoch, idx)
                exit(1)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        print("File ", i+1, " Done!")
        if (i+1) % 100 == 0:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

torch.save(net, "model")
