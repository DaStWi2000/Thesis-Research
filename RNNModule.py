import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNNModule(nn.Module):
    def __init__(self, cir_length, sym_blk):
        super(RNNModule,self).__init__()
        self.fc1 = nn.Linear(sym_blk*(1+2+2*cir_length)+2*cir_length,512)
        nn.init.normal_(self.fc1.weight,std=math.sqrt(2)/(sym_blk*(1+2+2*cir_length)+2*cir_length))
        self.fc2 = nn.Linear(512,256)
        nn.init.normal_(self.fc2.weight,std=math.sqrt(2)/512)
        self.fc3 = nn.Linear(256,128)
        nn.init.normal_(self.fc3.weight,std=math.sqrt(2)/256)
        self.fc4 = nn.Linear(128,sym_blk*2*cir_length)
        nn.init.normal_(self.fc4.weight,std=math.sqrt(2)/128)

    def forward(self,x,hidden):
        x = F.relu(self.fc1(torch.cat((x, hidden))))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
