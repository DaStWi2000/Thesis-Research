import torch
import torch.nn as nn
import torch.nn.functional as F

class ChEst(nn.Module):
    def __init__(self):
        super(ChEst,self).__init__()
        self.fc1 = nn.Linear(1+2+cir_length*2+cir_length*2,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,cir_length*2)

    def forward(self,x,hidden):
        x = F.relu(self.fc1(torch.cat((hidden, x))))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
