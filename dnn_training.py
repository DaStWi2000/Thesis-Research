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
net = ChEst(cir_length).to(dev)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters())
optimizer.zero_grad()
trainset = ChEstDataset(r"Dataset", SNR, False)
files_ind = list(range(0,len(trainset)))
random.shuffle(files_ind)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(files_ind,0):
        sample_indices = list(range(0,sym_blk*num_blk))
        hidden = torch.zeros((2*cir_length)).to(dev)
        act_channels = torch.zeros((num_blk*sym_blk,2*cir_length)).to(dev)
        pred_channels = torch.zeros((num_blk*sym_blk,2*cir_length)).to(dev)
        optimizer.zero_grad()
        sample = trainset[i]
        for idx in iter(sample_indices):
            usb, r, h_ls, h = torch.tensor(sample['usb'][idx]).to(dev), sample['r'][idx], sample['h_ls'][idx//sym_blk], sample['h'][idx]
            r = torch.tensor(numpy.concatenate((r.real,r.imag))).to(dev)
            h_ls = torch.tensor(numpy.concatenate((h_ls.real,h_ls.imag))).to(dev)
            h = torch.tensor(numpy.concatenate((h.real,h.imag))).to(dev)
            hidden = net(torch.cat((usb,r,h_ls)).float().to(dev),hidden)
            pred_channels[idx,:] = hidden
            act_channels[idx,:] = h
        loss = criterion(pred_channels, act_channels)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        print("File ", i+1, " Done!")
        if (i+1) % 100 == 0:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

torch.save(net, "model")
