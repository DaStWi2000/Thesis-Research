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
num_blk = math.floor(sr/sym_blk)

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
        y = mat_file['y']
        cirmat_ls = mat_file['cirmat_ls']
        cirmat = mat_file['cirmat']
        tx = tx[0:sym_blk*num_blk]
        tx = numpy.reshape(tx, (num_blk,sym_blk))
        y = y[200:]
        y = numpy.append(numpy.real(y),numpy.imag(y))
        rx = numpy.zeros((num_blk,sym_blk*2))
        for i in range(0,num_blk):
            rx[i,0:400] = y[i*400:(i+1)*400]
            rx[i,400:800] = y[(i+num_blk)*400:(i+1+num_blk)*400]
        cirmat_ls = numpy.append(numpy.real(cirmat_ls),numpy.imag(cirmat_ls),1)
        cirmat = cirmat[::sym_blk][:]
        if sr-sym_blk*num_blk > 0:
            cirmat = cirmat[1:]
        cirmat = numpy.append(numpy.real(cirmat),numpy.imag(cirmat),1)
        return (torch.Tensor(numpy.append(cirmat_ls,numpy.append(tx,rx,1),1)),torch.Tensor(cirmat))


class ChEst(nn.Module):
    def __init__(self):
        super(ChEst,self).__init__()
        self.fc1 = nn.Linear(2000,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,400)

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
        inputs, labels = data
        hidden = torch.zeros(400)
        optimizer.zero_grad()
        output = torch.zeros((12,400))
        for j in range(0,inputs[0].shape[0]):
            output[j][:] = net(inputs[0][j],hidden)
            hidden = output[j][:]
        loss = criterion(output, labels[0])
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

testset = ChEstDataset(r"Dataset", SNR, True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
num_bits = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        hidden = torch.zeros(400)
        output = torch.zeros((12,400))
        rx_sym = numpy.zeros(4800,'complex64')
        tx_sym = numpy.zeros(4800,'complex64')
        for i in range(0,num_blk):
            output[i][:] = net(inputs[0][i],output[i-1][:])
            rx_sym[i*400:(i+1)*400] = inputs[0][i][800:1200].numpy() + 1j*inputs[0][i][1200:1600].numpy()
            tx_sym[i*400:(i+1)*400] = inputs[0][i][400:800].numpy()
        output = output.numpy()
        output = output[:,0:200] + 1j*output[:,200:]
        conv_mat = numpy.zeros((4800,400),'complex64')
        tx_rec = numpy.zeros(4800,'complex64')
        for i in range(0, num_blk):
            conv_mat[i*400:round((i+.5)*400),:] = scipy.linalg.toeplitz(output[i,:],numpy.zeros(400))
            conv_mat[round((i+.5)*400):(i+1)*400,:] =  scipy.linalg.toeplitz(numpy.zeros(200),numpy.append(numpy.append(numpy.zeros(1),output[i,::-1]),numpy.zeros(199)))
            tx_rec[i*400:(i+1)*400] = spsolve(conv_mat[i*400:(i+1)*400],rx_sym[i*400:(i+1)*400])
        num_bits = num_bits + sum(abs(numpy.sign(numpy.real(tx_rec))-tx_sym))/2
        print(num_bits)
BER = math.log10(num_bits/(75*sym_blk*num_blk))
print(BER)
