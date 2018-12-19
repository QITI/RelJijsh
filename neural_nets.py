import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

				
# Neural Net
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(Net, self).__init__()
        self.input_layer=nn.Linear(input_size,hidden_size)
        self.hidden_layers=nn.ModuleList([nn.Linear(hidden_size,hidden_size) for i in range(num_hidden_layers)])
        self.output_layer=nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        out=self.input_layer(x)
        for layer in self.hidden_layers:
            out=layer(F.dropout(F.relu(out),p=0.05))
        out=self.output_layer(out)
        return out
		