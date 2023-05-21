import torch
import numpy as np
import d2lzh_pytorch as d2l

# y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + e

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# features x
features = torch.randn((n_train+n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2]
          + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_loss = []
    test_loss = []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_loss.append(loss(net(train_features), train_labels).item())
        test_loss.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_loss[-1], 'test_loss', test_loss[-1])
    d2l.semiplgy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss', 
                 range(1, num_epochs + 1), test_loss, ['train', 'test'], figsize=(10, 10))
    
    print('weight:', net.weight.data, 
          'bias:', net.bias.data)

# fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])
