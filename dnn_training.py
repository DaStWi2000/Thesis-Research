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

max_cirmat_r =  3.5659
min_cirmat_r = -2.8147
max_ls_r =  3.3768
min_ls_r = -2.5823
max_y_r =  5.7941
min_y_r = -5.8640
max_cirmat_i =  3.2713
min_cirmat_i = -2.9753
max_ls_i =  3.0532
min_ls_i = -2.4391
max_y_i =  5.6099
min_y_i = -5.6211

# Look into MATLAB DNN package to import Pytorch model
dev = ""
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
print(dev, "used")
net = ChEst(cir_length, sym_blk, num_blk).to(torch.double).to(dev)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters())
optimizer.zero_grad()
trainset = ChEstDataset(r"Dataset", SNR, False)
training_ind = list(range(0,round(0.7*len(trainset))))
random.shuffle(training_ind)
val_ind = list(range(round(0.7*len(trainset)),len(trainset)))
random.shuffle(val_ind)
val_loss = list()
for epoch in range(10):
    for i, data in enumerate(training_ind,0):
        est_channel = torch.zeros((sym_blk*num_blk,2*cir_length)).to(dev)
        act_channel = torch.zeros((sym_blk*num_blk,2*cir_length)).to(dev)
        optimizer.zero_grad()
        sample = trainset[i]
        usb, r, h_ls, h = torch.tensor(sample['usb'][0:num_blk*sym_blk]).to(dev), \
            sample['r'][0:num_blk*sym_blk], \
            sample['h_ls'], \
            sample['h'][0:num_blk*sym_blk]
        usb = torch.reshape((usb+1)/2, (num_blk,sym_blk))
        r = torch.reshape(torch.tensor(numpy.concatenate(((r.real-min_y_r)/(max_y_r-min_y_r),(r.imag-min_y_i)/(max_y_i-min_y_i)),axis=1)), (num_blk,sym_blk*2)).to(dev)
        h_ls = torch.tensor(numpy.concatenate(((h_ls.real-min_ls_r)/(max_ls_r-min_ls_r),(h_ls.imag-min_ls_i)/(max_ls_i-min_ls_i)),axis=1)).to(dev)
        h = torch.reshape(torch.tensor(numpy.concatenate(((h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r),(h.imag-min_cirmat_i)/(max_cirmat_i-min_cirmat_i)),axis=1)), (num_blk*sym_blk,cir_length*2)).to(dev)
        blk_input = torch.cat((usb,r,h_ls),axis=1)
        est_channel = net(blk_input)
        act_channel = h
        loss = criterion(est_channel, act_channel)
        if loss.item() != loss.item():
            print("ERROR", epoch, idx)
            exit(1)
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
        optimizer.step()
        print("File ", i+1, " Done! Loss: ", loss.item())
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(val_ind,0):
            est_channel = torch.zeros((sym_blk*num_blk,2*cir_length)).to(dev)
            act_channel = torch.zeros((sym_blk*num_blk,2*cir_length)).to(dev)
            optimizer.zero_grad()
            sample = trainset[i]
            usb, r, h_ls, h = torch.tensor(sample['usb'][0:num_blk*sym_blk]).to(dev), \
                sample['r'][0:num_blk*sym_blk], \
                sample['h_ls'], \
                sample['h'][0:num_blk*sym_blk]
            usb = torch.reshape((usb+1)/2, (num_blk,sym_blk))
            r = torch.reshape(torch.tensor(numpy.concatenate(((r.real-min_y_r)/(max_y_r-min_y_r),(r.imag-min_y_i)/(max_y_i-min_y_i)),axis=1)), (num_blk,sym_blk*2)).to(dev)
            h_ls = torch.tensor(numpy.concatenate(((h_ls.real-min_ls_r)/(max_ls_r-min_ls_r),(h_ls.imag-min_ls_i)/(max_ls_i-min_ls_i)),axis=1)).to(dev)
            h = torch.reshape(torch.tensor(numpy.concatenate(((h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r),(h.imag-min_cirmat_i)/(max_cirmat_i-min_cirmat_i)),axis=1)), (num_blk*sym_blk,cir_length*2)).to(dev)
            blk_input = torch.cat((usb,r,h_ls),axis=1)
            est_channel = net(blk_input)
            act_channel = h
            loss = criterion(est_channel, act_channel)
            running_loss += loss.item()
        val_loss.append(running_loss/len(val_ind))
    torch.save(net, "model"+str(epoch))
with file = open("losses.csv", "w"):
    for i in range(0,10):
        file.write(str(val_loss[i])+" ")
