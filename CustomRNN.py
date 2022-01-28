import torch
import torch.nn as nn
import math
from RNNModule import RNNModule

class CustomRNN(nn.Module):
    def __init__(self, cir_length, sym_blk, num_blk):
        super(CustomRNN,self).__init__()
        self.list = nn.ModuleList([RNNModule(cir_length, sym_blk) for i in range(0, num_blk)])
        self.num_blk = num_blk
        self.sym_blk = sym_blk
        self.cir_length = cir_length

    # Look at each layer and double check calculations
    def forward(self, input):
        x = torch.zeros((self.sym_blk*self.num_blk, self.cir_length*2))
        tmp = self.list[0](input[0,:],torch.zeros((self.sym_blk*self.cir_length*2)))
        x[0:self.sym_blk,:] = tmp.reshape((self.sym_blk,self.cir_length*2))
        for i in range(1, self.num_blk):
            tmp = self.list[i](input[i,:], tmp)
            x[i*self.sym_blk:(i+1)*self.sym_blk,:] = tmp.reshape((self.sym_blk,self.cir_length*2))
        return x
