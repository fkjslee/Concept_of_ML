# weight_decay
为了防止过拟合，限制$|\omega|$的值过大
原公式为:
设损失函数L是关于参数$\omega$的函数
$$ L=F(\omega) + \Omega(\omega)$$
其中, F是原本的损失函数, $\Omega$是惩罚项, 且
$$ \Omega=\frac{\alpha}{2}*|\omega|^2 $$
则:
$$ \frac{dL}{d\omega} $$
$$ =\frac{dF}{d\omega}+\frac{d\Omega}{d\omega} $$
$$ =\frac{dF}{d\omega}+\alpha\omega  $$

所以每次需要加入惩罚项的时候只需要在每次计算grad的时候加上$\alpha\omega$,
pytorch已经实现这个功能, 就是weight_decay.代表的就是这里的$\alpha$

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
