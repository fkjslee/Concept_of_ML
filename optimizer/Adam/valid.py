import torch
import math


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.w1 = torch.nn.Linear(1, 3)
        self.relu = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))


class MyAdam:
    def __init__(self, lr=1e-2, betas=(0.9, 0.999)):
        self.lr = lr
        self.betas = betas
        self.state_v = None  # v_t: union of g  v_0 = 0
        self.state_s = None  # s_t: union of g^2  s_0 = 0
        self.step = 0

    def get_res(self, p, g):
        if self.state_v is None:
            self.state_v = torch.zeros(p.shape)
            self.state_s = torch.zeros(p.shape)
        self.state_v = self.betas[0] * self.state_v + (1-self.betas[0]) * g
        self.state_s = self.betas[1] * self.state_s + (1-self.betas[1]) * (g ** 2)
        self.step += 1
        b1 = 1 - self.betas[0] ** self.step
        b2 = 1 - self.betas[1] ** self.step
        return p - self.lr / b1 * self.state_v / (torch.sqrt(self.state_s) / math.sqrt(b2) + 1e-8)


def get_train_data():
    x1 = torch.arange(4.5, 5, 0.5).reshape(-1, 1)
    y_real = x1 * x1 + 1
    return x1, y_real


config = {
    'seed': 0,
    'lr': 1,
    'momentum': 0.5,
    'epoch': 200,
    'eps': 1e-7,
    'dampening': 0,
    'weight_decay': 0,
    'beta1': 0.9,
    'beta2': 0.999,
}


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    torch.manual_seed(config['seed'])
    model = MyModule()
    optimizer_torch = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))
    optimizer_mine = MyAdam(lr=config['lr'], betas=(config['beta1'], config['beta2']))
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
        print(calc_by_mine, calc_by_torch)
        print(torch.abs(calc_by_torch - calc_by_mine) < config['eps'])
        assert(torch.all(torch.abs(calc_by_torch - calc_by_mine) < config['eps']))
