import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy
from ChEstDataset import ChEstDataset
import scipy
from CustomRNN import CustomRNN

#ADD JOSHUA'S AND THIS TO GITHUB REPO
#WORK ON README.MD

#CONSTANTS
SNR = 20
sr = 5000
cir_length = 2
sym_blk = cir_length*2
#num_blk = sr//sym_blk
num_blk = 5
num_epochs = 1000

#Scaling Factors
max_cirmat_r = 0.1708
min_cirmat_r = -0.1725
max_ls_r = 0.1693
min_ls_r = -0.1689
max_y_r = 0.2108
min_y_r = -0.2125
max_cirmat_i = 0.1441
min_cirmat_i = -0.1762
max_ls_i = 0.1419
min_ls_i = -0.1727
max_y_i = 0.1762
min_y_i = -0.1719

#Look into MATLAB DNN package to import Pytorch model
#Chooses CPU or GPU based on hardware
dev = ""
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
print(dev, "used")

#Initializes channel estimator of cir_length taps with sym_blk number of symbols
#tinynet = RNNModule(cir_length, sym_blk, num_blk).to(dev)
tinynet = CustomRNN(cir_length, sym_blk, num_blk).to(dev)
#Uses MSE (L2 Norm)
criterion = nn.MSELoss()
#Uses Adaptive Moment Estimation
optimizer = optim.Adam(tinynet.parameters(),lr=0.0001)
optimizer.zero_grad()

#Sets the training set to the SNR training set
trainset = ChEstDataset(r"Dataset", SNR, False)

#Divides the training dataset into training and validation sets
training_ind = list(range(0,round(0.7*len(trainset))))
val_ind = list(range(round(0.7*len(trainset)),len(trainset)))
#Shuffles the training and validation sets
random.shuffle(training_ind)
random.shuffle(val_ind)

#Initializes list that holds MSE of validation set at each epoch
val_loss = list()
batch_loss = list()

#Trains the Neural Network for num_epochs epochs
for epoch in range(num_epochs):
    #Training set
    for i, data in enumerate(training_ind,0):
        #Zeros out the gradiant of the optimizer
        optimizer.zero_grad()
        #Gets a random sample (set of blocks) and reads in the values
        sample = trainset[data]
        usb, r, h_ls, h = sample['usb'][0:num_blk*sym_blk], \
            sample['r'][0:num_blk*sym_blk], \
            sample['h_ls'][0:num_blk], \
            sample['h'][0:num_blk*sym_blk]
        #Reshapes the the transmitted bits so that there are num_blk rows and sym_blk columns
        usb = usb.reshape((num_blk,sym_blk))
        r = r.reshape((num_blk,sym_blk))
        #If the received signal has an imaginary component, this is taken into account
        if max_y_i-min_y_i > 0:
            r = (r.real-min_y_r)/(max_y_r-min_y_r)+1j*(r.imag-min_y_i)/(max_y_i-min_y_i)
            h_ls = (h_ls.real-min_ls_r)/(max_ls_r-min_ls_r)+1j*(h_ls.imag-min_ls_i)/(max_ls_i-min_ls_i)
            h = (h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r)+1j*(h.imag-min_cirmat_i)/(max_cirmat_i-min_cirmat_i)
        #Otherwise only the real components get resized
        else:
            r = (r.real-min_y_r)/(max_y_r-min_y_r)
            h_ls = (h_ls.real-min_ls_r)/(max_ls_r-min_ls_r)
            h = (h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r)
        #Concatenates all the columns together as input into the neural network
        blk_input = torch.tensor(numpy.concatenate((usb.real, r.real, h_ls.real, usb.imag, r.imag, h_ls.imag),axis=1)).float().to(dev)
        #Initializes estimated channel
        est_channel = torch.zeros((num_blk*sym_blk,cir_length*2))
        act_channel = torch.tensor(numpy.concatenate((h.real,h.imag),axis=1)).float().to(dev)
        # tmp = tinynet(blk_input[0,:], torch.zeros((sym_blk*cir_length*2)))
        # for k in range(0, num_blk-1):
        #     est_channel[k*sym_blk:(k+1)*sym_blk,:] = tmp.reshape((sym_blk,cir_length*2))
        #     tmp = tinynet(blk_input[k+1,:],tmp)
        # est_channel[(num_blk-1)*sym_blk:,:] = tmp.reshape((sym_blk,cir_length*2))
        est_channel = tinynet(blk_input)
        #Computes the loss, if error, the program exits
        loss = criterion(est_channel, act_channel)
        if loss.item() != loss.item():
            print("ERROR", epoch, idx)
            exit(1)
        #Descends the gradient
        loss.backward()
        #nn.utils.clip_grad_value_(tinynet.parameters(), clip_value=1.0)
        optimizer.step()
        #Updates status
        print("File ", i+1, " Done! Loss: ", loss.item())
        batch_loss.append(loss.item())
    #Validation set
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(val_ind):
            #Gets a random sample (set of blocks) and reads in the values
            sample = trainset[data]
            usb, r, h_ls, h = sample['usb'][0:num_blk*sym_blk], \
                sample['r'][0:num_blk*sym_blk], \
                sample['h_ls'][0:num_blk], \
                sample['h'][0:num_blk*sym_blk]
            #Reshapes the the transmitted bits so that there are num_blk rows and sym_blk columns
            usb = usb.reshape((num_blk,sym_blk))
            #If the received signal has an imaginary component, this is taken into account
            if max_y_i-min_y_i > 0:
                r = (r.real.reshape((num_blk,sym_blk))-min_y_r)/(max_y_r-min_y_r)+1j*(r.imag.reshape((num_blk,sym_blk))-min_y_i)/(max_y_i-min_y_i)
                h_ls = (h_ls.real-min_ls_r)/(max_ls_r-min_ls_r)+1j*(h_ls.imag-min_ls_i)/(max_ls_i-min_ls_i)
                h = (h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r)+1j*(h.imag-min_cirmat_i)/(max_cirmat_i-min_cirmat_i)
            #Otherwise only the real components get resized
            else:
                r = (r.real.reshape((num_blk,sym_blk))-min_y_r)/(max_y_r-min_y_r)
                h_ls = (h_ls.real-min_ls_r)/(max_ls_r-min_ls_r)
                h = (h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r)
            #Concatenates all the columns together as input into the neural network
            blk_input = torch.tensor(numpy.concatenate((usb.real, r.real, h_ls.real, usb.imag, r.imag, h_ls.imag),axis=1)).float().to(dev)
            #Initializes estimated channel
            est_channel = torch.zeros((num_blk*sym_blk,cir_length*2))
            #Gets the output of the neural network for this collection of blocks
            # tmp = tinynet(blk_input[0,:], torch.zeros((sym_blk*cir_length*2)))
            # for k in range(0, num_blk-1):
            #     est_channel[k*sym_blk:(k+1)*sym_blk,:] = tmp.reshape((sym_blk,cir_length*2))
            #     tmp = tinynet(blk_input[k+1,:],tmp)
            # est_channel[(num_blk-1)*sym_blk:,:] = tmp.reshape((sym_blk,cir_length*2))
            est_channel = tinynet(blk_input)
            #Converts the actual channel to a float
            act_channel = torch.tensor(numpy.concatenate((h.real,h.imag),axis=1)).float().to(dev)
            #Computes the loss
            loss = criterion(est_channel, act_channel)
            running_loss += loss.item()
        #Updates status
        val_loss.append(running_loss/len(val_ind))
    #Saves the current model
    torch.save(tinynet, "model"+str(epoch))
#Writes the losses to a file
with open("losses.csv", "w") as file:
    for item in val_loss:
        file.write(str(item)+",")
    file.write("\n")
    for item in batch_loss:
        file.write(str(item)+",")
