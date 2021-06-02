<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# SGD with momentum
## 公式
$$v_{t+1} = \mu * v_{t} + g_{t+1}$$
$$p_{t+1} = p_{t} - \text{lr} * v_{t+1}$$

## 参数
pytorch的SGD默认是momentum=0, dampening=0, weight_decay=0, 没有nesterov
### momentum
即这里的$\mu$
### dampening
SGD中的dampening参数, 即把公式改成, 但是$v_0$任然是$g_0$不变
$$v_{t+1} = \mu * v_{t} + (1-dampening)*g_{t+1}$$
$$p_{t+1} = p_{t} - \text{lr} * v_{t+1}$$
为什么弄成1-dampening而不是直接dampening? 不明白。
### weight_decay
在计算每步的grad的时候, g = g + weight_decay * p.
似乎是为了防止过拟合。不太懂.
### nesterov
在计算每步的更新步长的时候考虑当前状态的梯度grad
## 理解
$v_{t}$指的是到t时刻前面所有grad的联合，每次在前面grad的联合基础上加上当前grad.如果方向都统一可以增加步长，相反则减少步长。
### reference
> https://pytorch.org/docs/stable/optim.html?highlight=sgd#torch.optim.SGD

