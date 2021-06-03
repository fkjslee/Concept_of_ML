<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# RMSprop
## 公式
$$s_t=\gamma * s_{t-1}+(1 - \gamma) * g_t^2$$
$$p_t=p_{t-1} - \frac{lr}{\sqrt{s_t}+eps}*g_t$$

## 参数
pytorch的Adagrad默认weight_decay=0, eps=1e-8 都不需要考虑
### $\gamma$
grad联合成新的学习率时，前面的grad权重每次变为之前的$\gamma$倍, 当前grad的重要性为$(1-\gamma)$,  
通常取0.9, pytorch取的0.99
### momentum($\mu$)
即代码中的mu, 公式变成:
$$s_t=\gamma * s_{t-1}+(1 - \gamma) * g_t^2$$
$$v_t=\mu * v_{t-1} + \frac{g}{\sqrt{s_t}+eps}$$
$$p_t=p_{t-1} - lr * v_t $$
实际是之前用$\frac{lr}{\sqrt{s_t}+eps}*g_t$更新参数p, 改成了所有$\frac{lr}{\sqrt{s_t}+eps}*g_t$的联合更新p,
但似乎$v_t$会无限变大？
### centered
公式$p_t=p_{t-1} - \frac{lr}{\sqrt{s_t}+eps}*g_t$中的$s_t$变成了$s_t+M$, M是前面所有grad的联合(平方求和后开放,代码中未验证)
## 理解
修复了Adagrad后期学习率过小的问题


