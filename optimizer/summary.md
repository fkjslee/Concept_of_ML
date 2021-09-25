# still not know
最朴素的optimizer, 每次更新+lr*g

# SGD
每次更新量会参考之前g的联合

# Adagrad
统一学习率改成了每次参数各自学习率，通过统计之前grad的累计

# RMSprop
解决 Adagrad后期过大的问题，设立窗口

# Adam
联合SGD的累加g思想和RMSProp的累计g^2的思想，并加上了偏差修正

# AdamW
解决Adam的L2正则化问题，L2正则不适合Adam
