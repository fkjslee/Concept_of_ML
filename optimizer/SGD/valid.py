import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.w1 = torch.nn.Linear(1, 3)
        self.relu = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))


class MySGD:
    def __init__(self, mu, lr):
        self.mu = mu
        self.lr = lr
        self.v = None

    def get_res(self, p, g):  # where p is p_i, g is g{i+1}
        if self.v is None:
            self.v = g  # get v_0
        else:
            self.v = self.mu * self.v + g  # v_{i+1} = \mu * v_t + g{i+1}
        return p - self.lr * self.v  # p_{i+1} = p_i - \mu * v_{i+1}


def get_train_data():
    x1 = torch.arange(4.5, 5, 0.5).reshape(-1, 1)
    y_real = x1 * x1 + 1
    return x1, y_real


config = {
    'seed': 0,
    'lr': 1e-2,
    'momentum': 0.4,
    'epoch': 200,
    'eps': 1e-7,
}


if __name__ == "__main__":
    torch.manual_seed(config['seed'])
    model = MyModule()
    optimizer_torch = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    optimizer_mine = MySGD(config['momentum'], config['lr'])
    x1, y_real = get_train_data()
    for epoch in range(config['epoch']):
        y_pred = model(x1)
        loss = torch.nn.MSELoss()(y_pred, y_real)
        optimizer_torch.zero_grad()
        loss.backward()
        p = optimizer_torch.param_groups[0]['params'][0].clone().detach()  # w_0
        g = optimizer_torch.param_groups[0]['params'][0].grad.clone().detach()  # w_0' grad
        calc_by_mine = optimizer_mine.get_res(p, g)
        optimizer_torch.step()
        calc_by_torch = optimizer_torch.param_groups[0]['params'][0]
        print('*' * 50, 'check epoch = %d' % epoch, '*' * 50)
        # print(p)
        # print(g)
        # print(calc_by_mine, calc_by_torch)
        assert(torch.all(torch.abs(calc_by_torch - calc_by_mine) < config['eps']))
