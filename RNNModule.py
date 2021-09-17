import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModule(nn.Module):
    def __init__(self, cir_length):
        super(RNNModule,self).__init__()
        self.fc1 = nn.Linear(1+2+2*cir_length+2*cir_length,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,2*cir_length)

    def forward(self,x,hidden):
        x = F.relu(self.fc1(torch.cat((x, hidden))))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
