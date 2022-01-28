import torch
import torch.optim as optim
import random
import math
import numpy
from ChEstDataset import ChEstDataset
import scipy
from RNNModule import RNNModule

#ADD JOSHUA'S AND THIS TO GITHUB REPO
#WORK ON README.MD

#CONSTANTS
SNR = 20
sr = 5000
cir_length = 2
sym_blk = cir_length*2
# num_blk = sr//sym_blk
num_blk = 20

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

# Look into MATLAB DNN package to import Pytorch model
#Chooses CPU or GPU based on hardware
dev = ""
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
print(dev, "used")

#Loads in user specified model
tinynet = torch.load("model999")
test_loss = list()

#Sets the training set to the SNR testing set
testset = ChEstDataset(r"Dataset", SNR, False)
files_ind = list(range(round(0.7*len(testset)),len(testset)))
#Testing set
with torch.no_grad():
    for i, data in enumerate(files_ind):
        #Gets a sample (set of blocks) and reads in the values
        sample = testset[data]
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
            h = torch.tensor(numpy.concatenate(((h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r),(h.imag-min_cirmat_i)/(max_cirmat_i-min_cirmat_i)),axis=1)).to(dev)
        #Otherwise only the real components get resized
        else:
            r = (r.real.reshape((num_blk,sym_blk))-min_y_r)/(max_y_r-min_y_r)
            h_ls = (h_ls.real-min_ls_r)/(max_ls_r-min_ls_r)
            h = torch.tensor((h.real-min_cirmat_r)/(max_cirmat_r-min_cirmat_r),torch.zeros(h.size())).to(dev)
        #Concatenates all the columns together as input into the neural network
        blk_input = torch.tensor(numpy.concatenate((usb.real, r.real, h_ls.real, usb.imag, r.imag, h_ls.imag),axis=1)).float().to(dev)
        #Gets the output of the neural network for this collection of blocks
        est_channel = tinynet(blk_input)

        #Convert back into usable channel matrices
        est_channel = est_channel.to("cpu").numpy()
        act_channel = h.to("cpu").numpy()
        ls_channel = h_ls
        act_channel = act_channel[:,0:cir_length]*(max_cirmat_r-min_cirmat_r)+min_cirmat_r+1j*act_channel[:,cir_length:]*(max_cirmat_i-min_cirmat_i)+min_cirmat_i*1j
        est_channel = est_channel[:,0:cir_length]*(max_cirmat_r-min_cirmat_r)+min_cirmat_r+1j*est_channel[:,cir_length:]*(max_cirmat_i-min_cirmat_i)+min_cirmat_i*1j
        if max_ls_i-min_ls_i > 0:
            ls_channel = ls_channel.real*(max_ls_r-min_ls_r)+min_ls_r+1j*ls_channel.imag*(max_ls_i-min_ls_i)+min_ls_i*1j
        else:
            ls_channel = ls_channel.real*(max_ls_r-min_ls_r)+min_ls_r

        #Computes True MSE of ChEst
        test_loss.append((numpy.abs(numpy.square(est_channel-act_channel))).mean(axis=None))
        print("File ", i+1, " Done!")

#Saves all MSEs to CSV file
with open("val_loss.csv", "w") as file:
    for item in test_loss:
        file.write(str(item)+",")
