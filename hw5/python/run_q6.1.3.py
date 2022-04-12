import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import ReLU
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./CIFAR', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.001
hidden_size = 64
batch_num = 4

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet()
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

train_loss = []
train_acc = []

for itr in range(max_iters):

    total_loss = 0
    avg_acc = 0
    n_data = 0
    n_batches = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        xb, label = data
        pred = model(xb)
        
        loss = Loss(pred, label)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        _, pred_label = torch.max(pred, 1)
        
        n_data += label.size(0)
        n_batches += 1
        acc = (pred_label == label).sum().item()
        total_loss += loss.item()
        avg_acc += acc
    
    avg_acc /= n_data
    total_loss /= n_batches
    train_loss.append(total_loss)
    train_acc.append(avg_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

# plot loss curves
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.savefig('../output/q6.1.3-loss.png')
plt.show()
# plot accuracy curves
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.savefig('../output/q6.1.3-acc.png')
plt.show()
