
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,num):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.conv3=nn.Conv2d(16,32,4)
        self.dropout=nn.Dropout2d()
        self.fc1 = nn.Linear(32 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=F.relu(self.conv3(x))
        x=self.dropout(x)
        x=x.view(-1,32*10*10)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
