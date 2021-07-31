import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import scipy.linalg
import math
import numpy
import os
import time
from scipy.sparse.linalg import spsolve

#ADD JOSHUA'S AND THIS TO GITHUB REPO
#WORK ON README.MD

SNR = 20
sym_blk = 400
sr = 5000
num_files = 500
num_blk = math.floor(sr/sym_blk)
cir_length = 200

class ChEstDataset(torch.utils.data.Dataset):
    # Dataset for Channel Estimation Simulation
    def __init__(self, root_dir, SNR, test):
        self.mat_files = list()
        for file in os.listdir(root_dir):
            if test == True:
                if file.startswith("tv_test_"+str(SNR)):
                    self.mat_files.append(os.path.join(root_dir,file))
            else:
                if file.startswith("tv_"+str(SNR)):
                    self.mat_files.append(os.path.join(root_dir,file))

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        # Get working with csv/non-MAT file
        mat_file = scipy.io.loadmat(self.mat_files[idx])
        tx = mat_file['tx_symbols']
        tx = tx[0:sym_blk*num_blk]
        y = mat_file['y']
        y = y[0:sym_blk*num_blk]
        y = numpy.concatenate((y.real,y.imag),axis=1)
        cirmat_ls = mat_file['cirmat_ls']
        cirmat_ls = numpy.concatenate((cirmat_ls.real,cirmat_ls.imag),axis=0)
        cirmat = mat_file['cirmat']
        cirmat = cirmat[0:sym_blk*num_blk]
        cirmat = numpy.concatenate((cirmat.real,cirmat.imag),axis=1)
        sample = {'usb' : torch.tensor(tx), 'r' : torch.tensor(y), 'h_ls' : torch.tensor(cirmat_ls), 'h' : torch.tensor(cirmat)}
        return sample


class ChEst(nn.Module):
    def __init__(self):
        super(ChEst,self).__init__()
        self.fc1 = nn.Linear(1+2+2*cir_length+2*cir_length,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,cir_length*2)

    def forward(self,x,hidden):
        x = F.relu(self.fc1(torch.cat((hidden, x))))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = ChEst()
#net.load_state_dict(torch.load(r"C:\Users\dswil\Documents\UA Electrical Engineering Degree\THESIS RESEARCH\model"))
# Look into MATLAB DNN package to import Pytorch model

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters())
optimizer.zero_grad()
trainset = ChEstDataset(r"Dataset", SNR, False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad()
        hidden = torch.zeros((2*cir_length))
        output = torch.zeros((num_blk*sym_blk,2*cir_length))
        for j in range(0,num_blk*sym_blk):
            output[j][:] = net(torch.cat((data['usb'][0][j],data['r'][0][j],data['h_ls'][0][j//sym_blk],data['h_ls'][0][j//sym_blk+num_blk])).float(),hidden.float())
            hidden = output[j][:]
        loss = criterion(output, data['h'][0])
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        print("File ", i+1, " Done!")
        if (i+1) % 100 == 0:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

torch.save(net, "model")
# testset = ChEstDataset(r"Dataset", SNR, True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
# num_bits = 0
# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         hidden = torch.zeros(400)
#         output = torch.zeros((12,400))
#         rx_sym = numpy.zeros(4800,'complex64')
#         tx_sym = numpy.zeros(4800,'complex64')
#         for i in range(0,num_blk*sym_blk):
#             output[i][:] = net(inputs[0][i],output[i-1][:])
#             rx_sym[i*400:(i+1)*400] = inputs[0][i][800:1200].numpy() + 1j*inputs[0][i][1200:1600].numpy()
#             tx_sym[i*400:(i+1)*400] = inputs[0][i][400:800].numpy()
#         output = output.numpy()
#         output = output[:,0:200] + 1j*output[:,200:]
#         conv_mat = numpy.zeros((4800,400),'complex64')
#         tx_rec = numpy.zeros(4800,'complex64')
#         for i in range(0, num_blk):
#             conv_mat[i*400:round((i+.5)*400),:] = scipy.linalg.toeplitz(output[i,:],numpy.zeros(400))
#             conv_mat[round((i+.5)*400):(i+1)*400,:] =  scipy.linalg.toeplitz(numpy.zeros(200),numpy.append(numpy.append(numpy.zeros(1),output[i,::-1]),numpy.zeros(199)))
#             tx_rec[i*400:(i+1)*400] = spsolve(conv_mat[i*400:(i+1)*400],rx_sym[i*400:(i+1)*400])
#         num_bits = num_bits + sum(abs(numpy.sign(numpy.real(tx_rec))-tx_sym))/2
#         print(num_bits)
# BER = math.log10(num_bits/(75*sym_blk*num_blk))
# print(BER)
