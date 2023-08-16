import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

class FeedForwardNetNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=30)
        self.fc2 =nn.Linear(in_features=30, out_features=5)
        self.fc3 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))) )

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'

class Loader(Dataset):
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
xita=0.5
w=np.random.normal(loc=1,scale=1,size=(n,p))
u=np.random.normal(loc=1,scale=1,size=(n,p))
x=(w+xita*u)/(1+xita)
y=((2*x[:,1]-1)*(2*x[:,2]-1))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1)
min_max_scalerx = MinMaxScaler(feature_range=(0, 1))
min_max_scalery = MinMaxScaler(feature_range=(0, 1))
xtrain = min_max_scalerx.fit_transform(xtrain)
xtest =min_max_scalerx.transform(xtest)
ytrain =min_max_scalery.fit_transform(ytrain.reshape((-1,1)))
ytest =min_max_scalery.transform(ytest.reshape((-1,1)))


trainData=Loader(xtrain,ytrain)
trainData=DataLoader(trainData)
testData=Loader(xtest,ytest)
testData=DataLoader(testData)
input_size=x.shape[1]
num_epochs=200
model = FeedForwardNetNetwork(input_size=input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for data, labels in trainData:
        #print(data.shape,labels.shape)
        data = data.float().to(device=device)
        labels = labels.float().to(device=device)
        pred = model(data)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pred,true=[],[]
    with torch.no_grad():
        for data, labels in testData:

            data = data.float().to(device=device)
            labels = labels.float().to(device=device)
            pred_ = model(data)
            pred.append(pred_.detach().numpy()[0])
            true.append(labels.detach().numpy()[0])

    true,pred=np.array(true).reshape((-1,1)),np.array(pred).reshape((-1,1))
    print('the test score is {}'.format(r2_score(y_true=true,y_pred=pred) )   )

params=model.state_dict()

W=[params['fc1.weight'].numpy().T,params['fc2.weight'].numpy().T,params['fc3.weight'].numpy().T]
B=[params['fc1.bias'].numpy(),params['fc2.bias'].numpy(),params['fc3.bias'].numpy()]

L=3
# the forward pass can be computed as a sequence of matrix multiplications and nonlinearities
# return A as a list with length of layer number
A = [x]+[None]*L
for l in range(L):
    A[l+1] = np.maximum(0,A[l].dot(W[l])+B[l]  )


#create a list to store relevance scores at each layer
# return R as a list with length of layer number
T=y
R = [None]*L + [A[L]*T[:,None]  ]


def rho(w,l):  return w + [None,0.1,0.0,0.0][l] * np.maximum(0,w)
def incr(z,l): return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9

L=3
for l in range(1, L)[::-1]:
    w = rho(W[l], l)
    b = rho(B[l], l)

    z = incr(A[l].dot(w) + b, l)  # step 1
    s = R[l + 1] / z  # step 2
    c = s.dot(w.T)  # step 3
    R[l] = A[l] * c  # step 4


# propagate relevance scores until the pixels, we need to apply an alternate propagation rule that properly handles pixel values received as input
w  = W[0]
wp = np.maximum(0,w)
wm = np.minimum(0,w)
lb = A[0]*0-1
hb = A[0]*0+1

z = A[0].dot(w)-lb.dot(wp)-hb.dot(wm)+1e-9        # step 1
s = R[1]/z                                        # step 2
c,cp,cm  = s.dot(w.T),s.dot(wp.T),s.dot(wm.T)     # step 3
R[0] = A[0]*c-lb*cp-hb*cm
imp=np.mean(R[0],axis=0)
print(imp)