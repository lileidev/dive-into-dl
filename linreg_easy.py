import torch
import numpy as np
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


for X, y in data_iter:
    print(X, y)
    break

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

# net = LinearNet(num_inputs)
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
print(net)
print(type(net))

for param in net.parameters():
    print(param)

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)