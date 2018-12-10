import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

				
# Neural Net
class Net(nn.Module):
    def __init__(self, input_size, hidden_size,
                 output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.fc4 = nn.Linear(hidden_size, hidden_size)
        #self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out = self.fc1(x)
        #Repeated Sections
        out = self.fc2(F.dropout(F.relu(out), p=0.05))
        #out = self.fc4(F.dropout(F.relu(out), p=0.05))
        #out = self.fc5(F.dropout(F.relu(out), p=0.05))

        out = self.fc3(out)
        return out