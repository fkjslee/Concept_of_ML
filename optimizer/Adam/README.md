<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Adam
## 公式
$$v_t=\beta_1 * v_{t-1}+(1 - \beta_1) * g_t^2$$  
$$s_t=\beta_2 * s_{t-1}+(1 - \beta_2) * g_t^2$$  
$$ \hat{v_t}=\frac{v_t}{1-\beta_1^t} $$  
$$ \hat{s_t}=\frac{s_t}{1-\beta_2^t} $$  
$$ p = p-\frac{lr}{\sqrt{\hat{s_t}}+eps}*\hat{v_t}$$  
这里$v_t$代表$g_t$的联合  
$s_t$代表$g_t^2$的联合  
$\hat{v_t}, \hat{s_t}$是加了偏差修正  
$t$是当前迭代步数

## 参数
主要参数$beta_0$, $beta_1$
### $beta_0$, $beta_1$
分别是关于对于之前$g_t, g_t^2$的记忆量。另外涉及到偏差修正。  
$\hat{v_t}, \hat{s_t}$即对$v_t, s_t$的偏差修正  
举例说明, 如果不偏差修正, 当$\beta_1=0.9, $\beta_2=0.999, t=1$时  
$$ \frac{lr}{\sqrt{\hat{s_1}}+eps}\*\hat{v_1} $$  
$$ =\frac{lr}{\sqrt{s_1}+eps}\*v_1 $$  
$$ =\frac{lr}{\sqrt{\beta_2\*s_0+(1-\beta_2)\*g^2}+eps}\*(\beta_1\*v_0+(1-\beta_1)*g) $$  
$$ =\frac{lr}{\sqrt{0.001\*g^2}+eps}*0.1g $$
由于$v_1=0.1g, s_1=0.001g^2$, 参数的更新会很大程度的受到$s_1$的影响  
但如果加了偏差修正, $v_1=g, s_1=g^2$
$$ \frac{lr}{\sqrt{\hat{s_1}}+eps}\*\hat{v_1} $$  
$$ =\frac{lr}{\sqrt{g^2}+eps}\*g $$  
同理t=2时也有修正.


## 理解
加上了用$v_t$代表梯度g联合的思想(SGD)  
和用$s_t$代表$g^2$联合的思想(RMSProp)  
另外加入了偏差修正
