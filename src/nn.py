import torch
import torch.nn as nn
from collections import OrderedDict
from torch.optim.optimizer import Optimizer
import numpy as np

class SGLDOptim(Optimizer):
    def __init__(self, params, lr, weight_decay, data_size, cfg):
        self.batch_size = cfg['batch_size']
        self.data_size = data_size
        self.weight_decay = weight_decay
        defaults = dict(lr = lr)
        super(SGLDOptim, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                d_p = d_p.mul(self.data_size/(self.batch_size))
                d_p = d_p.add(param, alpha = self.weight_decay)
                param.add_(d_p, alpha = -lr/2)

                noise = torch.normal(0, 1, param.shape, device=param.device)
                param.add_(noise, alpha = np.sqrt(lr))

class NoisyNN(object):
    def __init__(self, cfg):
        nn_type = cfg['nn_type']
        ds_name = cfg['ds_name']
        if nn_type == 'test':
            self.nn = nn.Sequential(OrderedDict([
                                    ('line1', nn.Linear(1, 1)),
                                    ('Sigmoid', nn.Sigmoid()),
                                    ]))
        elif nn_type == 'LeNet5':
            assert ds_name == 'MNIST'
            channels = 1
            self.nn = nn.Sequential(OrderedDict([
                                    ('conv1', nn.Conv2d(channels, 6, 5)),
                                    ('relu1', nn.ReLU()),
                                    ('pool1', nn.MaxPool2d(2, 2)),
                                    ('conv2', nn.Conv2d(6, 16, 5)),
                                    ('relu2', nn.ReLU()),
                                    ('pool2', nn.MaxPool2d(2, 2)),
                                    ('conv3', nn.Conv2d(in_channels = 16,
                                                        out_channels = 120,
                                                        kernel_size = 5)),
                                    ('flatn', nn.Flatten()),
                                    ('relu3', nn.ReLU()),
                                    ('line4', nn.Linear(120, 84)),
                                    ('relu4', nn.ReLU()),
                                    ('line5', nn.Linear(84, 10)),
                                    ('softm', nn.LogSoftmax(dim = -1))
                                    ]))
        else:
            raise NotImplementedError(str(nn_type) +
                                      " model is not implemented")

    def saveWeights(self, path):
        torch.save(self.nn.state_dict(), path)

    def loadWeights(self, path):
        self.nn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    cfg = {'nn_type' : 'test',
           'ds_name' : None,
           'batch_size' : 4}
    test_network = NoisyNN(cfg)
    lr = 0.01
    optimizer = SGLDOptim(test_network.nn.parameters(), lr, 1, 10, cfg)
    criterion = nn.BCELoss(reduction = "sum")

    features = torch.tensor([[0],[0],[1],[1]], dtype = torch.float)
    labels = torch.tensor([0, 0, 1, 1], dtype = torch.float)
    T = 10000
    print(features)
    print(labels)
    loss_arr = []
    pbias = []
    print(test_network.nn)
    for t in range(T):
        optimizer.zero_grad()
        pred = test_network.nn(features)
        loss = criterion(pred.reshape(pred.shape[0]), labels)
        loss_arr.append(loss.detach().item())
        loss.backward()
        optimizer.step()
        if t > 8000:
            pbias.append(test_network.nn.line1.bias.detach().item())
    plt.hist(pbias, 100)
    pbias = np.array(pbias)
    print(np.std(pbias), np.mean(pbias), np.sqrt(lr))
    plt.show()
