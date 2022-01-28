import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNNModule(nn.Module):
    def __init__(self, cir_length, sym_blk):
        super(RNNModule,self).__init__()
        # fc1_in = sym_blk*(2+2+2*cir_length)+2*cir_length
        # fc4_out = sym_blk*2*cir_length
        # fc2_in = round(2*fc1_in/3+fc4_out)
        # fc3_in = round(2*fc2_in/3+fc4_out)
        # fc4_in = round(2*fc3_in/3+fc4_out)
        fc1_in = sym_blk*(2+2+2*cir_length)+2*cir_length
        fc2_in = 512
        fc3_in = 256
        fc4_in = 128
        fc4_out = sym_blk*2*cir_length
        self.fc1 = nn.Linear(fc1_in,fc2_in)
        nn.init.normal_(self.fc1.weight,std=math.sqrt(2)/fc1_in)
        self.fc2 = nn.Linear(fc2_in, fc3_in)
        nn.init.normal_(self.fc2.weight,std=math.sqrt(2)/fc2_in)
        self.fc3 = nn.Linear(fc3_in,fc4_in)
        nn.init.normal_(self.fc3.weight,std=math.sqrt(2)/fc3_in)
        self.fc4 = nn.Linear(fc4_in, fc4_out)
        nn.init.normal_(self.fc4.weight,std=math.sqrt(2)/fc4_in)

    # Look at each layer and double check calculations
    def forward(self,x,hidden):
        x = F.relu(self.fc1(torch.cat((x, hidden))))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
