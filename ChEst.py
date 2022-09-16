import torch
import torch.nn as nn
from RNNModule import RNNModule

class ChEst(nn.Module):
    def __init__(self, cir_length, sym_blk, num_blk):
        super(ChEst,self).__init__()
        template = RNNModule(cir_length, sym_blk)
        self.chest_network = nn.ModuleList([template for i in range(0,sym_blk)])
        self.row_num = num_blk
        self.group_size = sym_blk
        self.col_num = 2*cir_length

    def forward(self,x):
        est_channel = torch.zeros((self.row_num*self.group_size,self.col_num))
        output_h = self.chest_network[0](x[0,:],torch.zeros(self.col_num*self.group_size))
        est_channel[0:self.group_size,:] = torch.reshape(output_h, (self.group_size,self.col_num))
        for i in range(1, self.row_num):
            output_h = self.chest_network[i](x[i,:],output_h)
            est_channel[i*self.group_size:(i+1)*self.group_size:,:] = torch.reshape(output_h, (self.group_size,self.col_num))
        return est_channel
