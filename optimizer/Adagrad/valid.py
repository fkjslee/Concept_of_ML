import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.w1 = torch.nn.Linear(1, 3)
        self.relu = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))


class MyAdagrad:
    def __init__(self, mu, lr, dampening=0, weight_decay=0):
        self.mu = mu
        self.lr = lr
        self.v = None
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.state_sum = None

    def get_res(self, p, g):
        clr = self.lr
        if self.state_sum is None:
            self.state_sum = torch.zeros(p.shape)
        self.state_sum = self.state_sum + g * g
        std = self.state_sum.sqrt() + 1e-10
        p = p - clr * g / std
        return p


def get_train_data():
    x1 = torch.arange(4.5, 5, 0.5).reshape(-1, 1)
    y_real = x1 * x1 + 1
    return x1, y_real


config = {
    'seed': 0,
    'lr': 1e-4,
    'momentum': 0.5,
    'epoch': 200,
    'eps': 1e-7,
    'dampening': 0,
    'weight_decay': 0,
}


if __name__ == "__main__":
    torch.manual_seed(config['seed'])
    model = MyModule()
    optimizer_torch = torch.optim.Adagrad(model.parameters(), lr=config['lr'])
    optimizer_mine = MyAdagrad(config['momentum'], config['lr'], dampening=config['dampening'], weight_decay=config['weight_decay'])
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
