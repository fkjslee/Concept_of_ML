<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Adagrad
## 公式
$$s_t=s_{t-1}+g_t*g_t$$
$$p_t=p_{t-1} - \frac{lr}{\sqrt{s_t}+eps}*g_t$$

## 参数
pytorch的Adagrad默认lr_decay=0, weight_decay=0, eps=1e-10 都不需要考虑
### weight_decay
在计算每步的grad的时候, g = g + weight_decay * p.
似乎是为了防止过拟合。不太懂.
### initial_accumulator_value
指s_0的初始值 默认是0
## 理解
更新时候的统一学习率变成了每个参数都有各自的学习率(公式中的lr/sqrt(s_t))
另外，由于s_t会越来越大 会导致学习率变小(s_t是分母)

