<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# SGD with momentum
## 公式
$$v_{t+1} = \mu * v_{t} + g_{t+1}$$
$$p_{t+1} = p_{t} - lr * v_{t+1}$$

## 参数
pytorch的SGD默认是momentum=0, dampening=0, weight_decay=0, 没有nesterov
### momentum
即这里的$\mu$
### dampening
SGD中的dampening参数, 即把公式改成, 但是$v_0$任然是$g_0$不变
$$v_{t+1} = \mu * v_{t} + (1-dampening)*g_{t+1}$$
$$p_{t+1} = p_{t} - \text{lr} * v_{t+1}$$
为什么弄成1-dampening而不是直接dampening? 不明白。
### nesterov
在计算每步的更新步长的时候考虑当前状态的梯度grad
## 理解
更新时候由普通的梯度变成了梯度的联合$v_{t+1}$
### reference
> https://pytorch.org/docs/stable/optim.html?highlight=sgd#torch.optim.SGD

