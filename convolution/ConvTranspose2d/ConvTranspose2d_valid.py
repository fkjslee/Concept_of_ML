import cv2
import torch.nn as nn
import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.ConvTranspose2d(1, 1, (3, 3))


    def forward(self, x):
        x = self.layer1(x)
        return x


torch.manual_seed(0)
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True).reshape(1, 1, 2, 2)
model = MyModule()
y = model(x)

state_dict = model.state_dict()
kernel_w = state_dict['layer1.weight']
kernel_w = kernel_w[0, 0]
kernel_b = state_dict['layer1.bias']

x = x[0, 0]
dst_size = (4, 4)
src_size = (2, 2)
kernel_size = (3, 3)
trans = torch.zeros(dst_size[0] * dst_size[1], src_size[0] * src_size[1], dtype=torch.float32)
trans[0:3, 0] = kernel_w[0]
trans[4:7, 0] = kernel_w[1]
trans[8:11, 0] = kernel_w[2]
trans[1:4, 1] = kernel_w[0]
trans[5:8, 1] = kernel_w[1]
trans[9:12, 1] = kernel_w[2]
trans[4:7, 2] = kernel_w[0]
trans[8:11, 2] = kernel_w[1]
trans[12:15, 2] = kernel_w[2]
trans[5:8, 3] = kernel_w[0]
trans[9:12, 3] = kernel_w[1]
trans[13:16, 3] = kernel_w[2]
res = torch.matmul(trans, x.reshape(4, 1)) + kernel_b
assert torch.all(torch.abs(res.reshape(4, 4) - y) < 1e-7)
