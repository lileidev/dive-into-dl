import torch
import numpy as np
import random
import d2lzh_pytorch

# genearate dataset
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs))).to(torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

batch_size = 10
for x, y in d2lzh_pytorch.data_iter(batch_size, features, labels):
    print(x, y)
    break

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(True)
b.requires_grad_(True)

lr = 0.03
num_epochs = 3
net = d2lzh_pytorch.linreg
loss = d2lzh_pytorch.squared_loss

for epoch in range(num_epochs):
    for X, y in d2lzh_pytorch.data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        d2lzh_pytorch.sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch+1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
