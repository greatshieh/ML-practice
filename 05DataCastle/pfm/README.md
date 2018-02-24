## features.py


> 采用logistic回归实现
>
>数值型变量，包括连续变量和有序分类变量，进行 >**MinMaxScaler** 归一化。<br>
>所有字符型变量创建虚拟变量。<br>
>
>最优化模型参数:<br>
>{'C': 1, 'penalty': 'l1', 'random_state': 1, 'tol': >1e-06}
>
>得分 **0.89\*\*7**
<br>

> 不进行归一化<br>
> {'C': 1, 'penalty': 'l1', 'random_state': 2, 'tol': 0.0001}<br>
> 得分 **0.90\*\*\*7**
