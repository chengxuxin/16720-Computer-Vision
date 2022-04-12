import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import torch
from torch.nn import ReLU
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.005
hidden_size = 64
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

N, D = train_x.shape
N_output = train_y.shape[1]
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.act = ReLU()
        self.conv1 = torch.nn.Conv2d(1,16,5,1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(16,8,3,1)
        self.fc_input_dim = 8*6*6
        self.fc1 = torch.nn.Linear(self.fc_input_dim, 64)
        self.fc2 = torch.nn.Linear(64, N_output)
        self.last_act = torch.nn.Softmax()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, self.fc_input_dim)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        # x = self.last_act(x)
        return x

model = ConvNet()
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
val_x = torch.from_numpy(valid_x).float().reshape(-1,1,32,32)
val_y = torch.from_numpy(valid_y).int()
val_label = torch.nonzero(val_y, as_tuple=True)[1]

for itr in range(max_iters):
    val_pred = model(val_x)
    loss = Loss(val_pred, val_label)
    _, val_pred_label = torch.max(val_pred, 1)
    acc = (val_pred_label == val_label).sum().item()
    valid_loss.append(loss.item())
    valid_acc.append(acc/val_x.shape[0])

    total_loss = 0
    avg_acc = 0
    for xb, yb in batches:
        xb = torch.from_numpy(xb).float().reshape(-1,1,32,32)
        yb = torch.from_numpy(yb).int()
        label = torch.nonzero(yb.detach(), as_tuple=True)[1]
        
        pred = model(xb)
        
        loss = Loss(pred, label)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        _, pred_label = torch.max(pred, 1)

        acc = (pred_label == label).sum().item()
        total_loss += loss.item()
        avg_acc += acc
    
    avg_acc /= N
    total_loss /= batch_num
    train_loss.append(total_loss)
    train_acc.append(avg_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

# plot loss curves
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(valid_loss)), valid_loss, label="validation")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.savefig('../output/q6.1.2-loss.png')
plt.show()
# plot accuracy curves
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(valid_acc)), valid_acc, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.savefig('../output/q6.1.2-acc.png')
plt.show()
