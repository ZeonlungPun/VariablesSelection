import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyLoader(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __getitem__(self,index):
        data=self.data[index]
        labels=self.label[index]
        return data,labels
    def __len__(self):
        return len(self.data)


n=200
p=50
xita=0.25
w=np.random.normal(loc=1,scale=1,size=(n,p))
u=np.random.normal(loc=1,scale=1,size=(n,p))
x=(w+xita*u)/(1+xita)
y=((2*x[:,1]-1)*(2*x[:,2]-1)).reshape((-1,1))

Data=MyLoader(x,y)
Data=DataLoader(Data)
input_size=x.shape[1]
num_epochs=10
model = NeuralNetwork(input_size=input_size, num_classes=1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params= model.parameters(), lr=0.05)

for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(Data):
        data = data.float().to(device=device)
        data =torch.tensor(data,requires_grad=True)
        labels = labels.float().to(device=device)
        labels=torch.tensor(labels,requires_grad=True)

        scores = model(data)


        loss = criterion(scores, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



xx=torch.tensor(x,requires_grad=True).float().to(device=device)
for a in model.parameters():
    a.detach()
for batch_idx, (data, labels) in enumerate(Data):
    data = data.float().to(device=device)
    y=model(data)
    y.backward(torch.ones_like(y))
    print(data.grad)









