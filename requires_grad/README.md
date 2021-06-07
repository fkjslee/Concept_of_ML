# requires_grad

设置为True表示需要计算梯度  
False表示不需要计算梯度

# 例子
比如一个计算图长这样：  
![图片](https://github.com/fkjslee/github_image/blob/main/pic1.png)  
backward之后求出梯度长这样：  
![图片](https://github.com/fkjslee/github_image/blob/main/pic2.png)  
所以求出来$\frac{\partial y}{\partial a}=6a^2$

但如果取消w的梯度
即让
```
w.requires_read = False
```
那么图片变成了这样：
![图片](https://github.com/fkjslee/github_image/blob/main/pic3.png)

所以求出来$\frac{\partial y}{\partial a}=2*w$

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>