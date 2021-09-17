import torch
import torch.nn as nn
from RNNModule import RNNModule

class ChEst(nn.Module):
    def __init__(self, cir_length, sym_blk):
        super(ChEst,self).__init__()
        template = RNNModule(cir_length)
        self.chest_network = nn.ModuleList([template for i in range(0,sym_blk)])
        self.row_num = sym_blk
        self.col_num = 2*cir_length

    def forward(self,x):
        est_channel = torch.zeros((self.row_num,self.col_num))
        est_channel[0,:] = self.chest_network[0](x[0,:],torch.zeros(self.col_num))
        for i in range(1, self.row_num):
            est_channel[i,:] = self.chest_network[i](x[i,:],est_channel[i-1,:])
        return est_channel
