# 第4章 凸优化算法

## 本章导读

掌握了凸优化的理论基础后，本章将介绍求解凸优化问题的实用算法。从经典的一阶方法到二阶方法，从确定性算法到随机算法，我们将系统学习各类算法的原理、收敛性分析和实现技巧。通过本章学习，你将能够针对不同类型的凸优化问题选择合适的算法，并理解算法参数调优的基本原则。

---

## 4.1 无约束凸优化算法

### 4.1.1 梯度下降法

**基本思想**

梯度下降法是最基本的一阶优化算法。其核心思想是：在当前点沿目标函数的负梯度方向移动，因为负梯度方向是函数值下降最快的方向。

**算法描述**

算法 4.1（梯度下降法）

给定初始点 $x_0$，迭代直到收敛：
$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

其中 $\alpha_k > 0$ 是步长（学习率）。

**步长选择策略**

1. **固定步长**：$\alpha_k = \alpha$（常数）
   - 简单但可能收敛慢或发散
   - 需要满足 $\alpha < 2/L$，其中 $L$ 是 Lipschitz 常数

2. **精确线搜索**：
   $$\alpha_k = \arg\min_{\alpha \geq 0} f(x_k - \alpha \nabla f(x_k))$$
   - 理论上最优但计算代价高

3. **回溯线搜索（Backtracking line search）**：
   初始化 $\alpha = \alpha_0$，重复 $\alpha \leftarrow \beta \alpha$ 直到：
   $$f(x_k - \alpha \nabla f(x_k)) \leq f(x_k) - c \alpha \|\nabla f(x_k)\|_2^2$$
   其中 $\beta \in (0, 1)$，$c \in (0, 1/2)$（通常取 $c = 0.5$，$\beta = 0.8$）

**收敛性分析**

假设 $f$ 是凸函数且梯度 Lipschitz 连续，即：
$$\|\nabla f(x) - \nabla f(y)\|_2 \leq L \|x - y\|_2$$

定理 4.1 若 $f$ 是凸函数且 $\nabla f$ 是 $L$-Lipschitz 连续，使用固定步长 $\alpha \leq 1/L$，则：
$$f(x_k) - f(x^*) \leq \frac{\|x_0 - x^*\|_2^2}{2\alpha k}$$

即收敛速度为 $O(1/k)$。

若 $f$ 还是 $\mu$-强凸的（$\mu > 0$），则：
$$\|x_k - x^*\|_2^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|x_0 - x^*\|_2^2$$

即线性收敛（指数收敛）。

**Python 实现**

```python
import numpy as np

def gradient_descent(f, grad_f, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    """
    梯度下降法
    f: 目标函数
    grad_f: 梯度函数
    x0: 初始点
    """
    x = x0.copy()
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
    return x, k

def backtracking_line_search(f, grad_f, x, d, alpha=1.0, beta=0.8, c=0.5):
    """
    回溯线搜索
    d: 搜索方向（通常为 -grad_f(x)）
    """
    grad = grad_f(x)
    while f(x + alpha * d) > f(x) + c * alpha * np.dot(grad, d):
        alpha *= beta
    return alpha
```

### 4.1.2 随机梯度下降法

**动机**

对于大规模问题，计算完整梯度 $\nabla f(x)$ 代价很高。随机梯度下降（SGD）使用梯度的无偏估计来降低计算复杂度。

**基本形式**

若目标函数是有限和形式：
$$f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)$$

则完整梯度为：
$$\nabla f(x) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(x)$$

SGD 每次随机选择一个样本 $i_k$，使用 $\nabla f_{i_k}(x_k)$ 作为梯度的估计：
$$x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k)$$

**算法描述**

算法 4.2（随机梯度下降法）

重复直到收敛：
1. 随机选择 $i_k \in \{1, \ldots, n\}$
2. $x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k)$

**收敛性**

定理 4.2 若 $f$ 是凸函数，$\mathbb{E}[\|\nabla f_i(x)\|_2^2] \leq G^2$，使用递减步长 $\alpha_k = O(1/\sqrt{k})$，则：
$$\mathbb{E}[f(\bar{x}_k)] - f(x^*) \leq O(1/\sqrt{k})$$

其中 $\bar{x}_k = \frac{1}{k}\sum_{i=1}^k x_i$ 是平均迭代点。

**变体**

1. **小批量 SGD（Mini-batch SGD）**：每次使用 $m$ 个样本的梯度平均
2. **带动量的 SGD**：
   $$v_{k+1} = \beta v_k + \nabla f_{i_k}(x_k)$$
   $$x_{k+1} = x_k - \alpha_k v_{k+1}$$
3. **AdaGrad**：自适应调整每个坐标的学习率
4. **Adam**：结合动量和自适应学习率

### 4.1.3 牛顿法

**基本思想**

牛顿法利用二阶信息（Hessian 矩阵）来加速收敛。它通过在当前点用二次函数近似目标函数，然后直接跳到近似函数的极小点。

**算法推导**

在 $x_k$ 处的二阶泰勒展开：
$$f(x) \approx f(x_k) + \nabla f(x_k)^T(x - x_k) + \frac{1}{2}(x - x_k)^T \nabla^2 f(x_k)(x - x_k)$$

令梯度为零，得到牛顿步：
$$\nabla^2 f(x_k)(x_{k+1} - x_k) = -\nabla f(x_k)$$

即：
$$x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

**算法描述**

算法 4.3（牛顿法）

给定初始点 $x_0$，迭代直到收敛：
1. 计算梯度 $g_k = \nabla f(x_k)$ 和 Hessian $H_k = \nabla^2 f(x_k)$
2. 求解线性系统：$H_k d_k = -g_k$
3. $x_{k+1} = x_k + d_k$（或使用线搜索确定步长）

**收敛性**

定理 4.3（牛顿法的局部二次收敛） 设 $f$ 二阶连续可微，$x^*$ 是局部最优解，$\nabla^2 f(x^*) \succ 0$。若初始点 $x_0$ 足够接近 $x^*$，则牛顿法二次收敛：
$$\|x_{k+1} - x^*\|_2 \leq C \|x_k - x^*\|_2^2$$

**优缺点**

优点：
- 收敛速度快（二次收敛）
- 对条件数不敏感

缺点：
- 需要计算和存储 Hessian 矩阵（$O(n^2)$ 存储）
- 需要求解线性系统（$O(n^3)$ 计算）
- 可能收敛到鞍点或极大值点

### 4.1.4 拟牛顿法

**动机**

克服牛顿法计算 Hessian 的代价，同时保持较快的收敛速度。

**基本思想**

用近似矩阵 $B_k$（或 $H_k = B_k^{-1}$）代替 Hessian，满足**割线条件**（secant condition）：
$$B_{k+1}(x_{k+1} - x_k) = \nabla f(x_{k+1}) - \nabla f(x_k)$$

或记 $s_k = x_{k+1} - x_k$，$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$：
$$B_{k+1} s_k = y_k$$

**BFGS 算法**

BFGS（Broyden-Fletcher-Goldfarb-Shanno）是最流行的拟牛顿法。

更新公式：
$$B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$$

或使用逆 Hessian 近似 $H_k = B_k^{-1}$：
$$H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}$$

**算法描述**

算法 4.4（BFGS）

给定初始点 $x_0$ 和初始矩阵 $H_0$（通常 $H_0 = I$），迭代直到收敛：
1. 计算搜索方向：$d_k = -H_k \nabla f(x_k)$
2. 线搜索确定步长 $\alpha_k$
3. 更新：$x_{k+1} = x_k + \alpha_k d_k$
4. 计算 $s_k = x_{k+1} - x_k$，$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$
5. 更新 $H_{k+1}$（使用 BFGS 公式）

**收敛性**

BFGS 具有超线性收敛速度：
$$\lim_{k \to \infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} = 0$$

**L-BFGS**

对于大规模问题，存储 $H_k$ 仍然代价高昂。L-BFGS（Limited-memory BFGS）只保存最近的 $m$ 对 $(s_k, y_k)$，通过递归公式计算 $H_k \nabla f(x_k)$ 而不显式存储 $H_k$。

### 4.1.5 共轭梯度法

**基本思想**

共轭梯度法（Conjugate Gradient, CG）是求解大规模对称正定线性系统的有效方法，也可以用于优化问题。

**线性 CG**

求解 $Ax = b$，其中 $A \succ 0$。

算法 4.5（线性共轭梯度法）

初始化 $x_0$，$r_0 = b - Ax_0$，$p_0 = r_0$

对于 $k = 0, 1, \ldots$：
1. $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$
2. $x_{k+1} = x_k + \alpha_k p_k$
3. $r_{k+1} = r_k - \alpha_k A p_k$
4. $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$
5. $p_{k+1} = r_{k+1} + \beta_k p_k$

**非线性 CG**

用于一般优化问题，使用线搜索代替精确最小化。

**收敛性**

对于 $n$ 维二次问题，CG 最多 $n$ 步收敛（理论上）。对于一般凸函数，具有超线性收敛速度。

---

## 4.2 约束凸优化算法

### 4.2.1 投影梯度下降法

**基本思想**

对于约束优化问题 $\min_{x \in C} f(x)$，投影梯度下降法在每一步梯度下降后将结果投影回可行集 $C$。

**投影算子**

定义 4.1（投影算子） 集合 $C$ 上的投影算子定义为：
$$\Pi_C(x) = \arg\min_{y \in C} \|y - x\|_2$$

对于常见集合，投影有解析解：
- 非负象限：$\Pi_{\mathbb{R}_+^n}(x) = \max(x, 0)$（分量-wise）
- 盒约束：$\Pi_{[l, u]}(x) = \min(\max(x, l), u)$
- $l_2$ 球：$\Pi_{B}(x) = \frac{x}{\max(1, \|x\|_2/r)}$

**算法描述**

算法 4.6（投影梯度下降法）

给定初始点 $x_0 \in C$，迭代直到收敛：
$$x_{k+1} = \Pi_C(x_k - \alpha_k \nabla f(x_k))$$

**收敛性**

定理 4.4 若 $f$ 是凸函数且 $\nabla f$ 是 $L$-Lipschitz 连续，使用固定步长 $\alpha \leq 1/L$，则：
$$f(x_k) - f(x^*) \leq \frac{\|x_0 - x^*\|_2^2}{2\alpha k}$$

收敛速度与无约束梯度下降相同。

### 4.2.2 内点法

**基本思想**

内点法通过引入障碍函数将不等式约束问题转化为一系列无约束（或等式约束）问题，从可行域内部逼近最优解。

**障碍函数**

对于问题：
$$\min_x f_0(x) \quad \text{s.t.} \quad f_i(x) \leq 0, \quad Ax = b$$

引入对数障碍函数：
$$\phi(x) = -\sum_{i=1}^m \log(-f_i(x))$$

定义域为 $\{x \mid f_i(x) < 0, i = 1, \ldots, m\}$（严格可行集）。

**中心路径**

考虑参数化问题：
$$\min_x t f_0(x) + \phi(x) \quad \text{s.t.} \quad Ax = b$$

其中 $t > 0$ 是参数。当 $t \to \infty$，解趋向原问题最优解。

定义 4.2（中心路径） **中心路径**定义为：
$$x^*(t) = \arg\min_x \{t f_0(x) + \phi(x) \mid Ax = b\}, \quad t > 0$$

**算法描述**

算法 4.7（障碍法/内点法）

给定严格可行初始点 $x$，$t = t_0 > 0$，$\mu > 1$，容差 $\epsilon > 0$

重复直到 $m/t < \epsilon$：
1. **中心步**：从 $x$ 出发，求解 $\min_x t f_0(x) + \phi(x)$ s.t. $Ax = b$，得到 $x^*(t)$
2. $x \leftarrow x^*(t)$
3. **增加 $t$**：$t \leftarrow \mu t$

**收敛性**

定理 4.5 障碍法在 $O(\sqrt{m} \log(m/(t_0 \epsilon)))$ 次外迭代内收敛到 $\epsilon$-最优解。

每次中心步可以使用牛顿法求解，通常只需要少量牛顿步。

**原对偶内点法**

更高效的变体，同时更新原变量和对偶变量，通常只需要一次牛顿步 per 外迭代。

### 4.2.3 增广拉格朗日方法

已在第3章介绍，这里补充其作为约束优化算法的视角。

对于等式约束问题：
$$\min_x f(x) \quad \text{s.t.} \quad Ax = b$$

增广拉格朗日函数：
$$L_\rho(x, \nu) = f(x) + \nu^T(Ax - b) + \frac{\rho}{2}\|Ax - b\|_2^2$$

算法迭代：
1. $x^{k+1} = \arg\min_x L_\rho(x, \nu^k)$
2. $\nu^{k+1} = \nu^k + \rho(Ax^{k+1} - b)$

**收敛性**

对于凸问题，增广拉格朗日法收敛到最优解。收敛速度取决于 $\rho$ 的选择。

---

## 4.3 算法收敛性分析

### 4.3.1 收敛速度的定义

定义 4.3（收敛速度） 设序列 $\{x_k\}$ 收敛到 $x^*$：

- **Q-线性收敛**：$\lim_{k \to \infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} = \mu < 1$
- **Q-超线性收敛**：$\lim_{k \to \infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} = 0$
- **Q-二次收敛**：$\lim_{k \to \infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|^2} = \mu < \infty$
- **次线性收敛**：$O(1/k)$ 或 $O(1/\sqrt{k})$

### 4.3.2 一阶方法与二阶方法比较

| 特性 | 一阶方法 | 二阶方法 |
|------|----------|----------|
| 每次迭代计算量 | $O(n)$ | $O(n^2)$ 或 $O(n^3)$ |
| 存储需求 | $O(n)$ | $O(n^2)$ |
| 收敛速度 | 次线性/线性 | 超线性/二次 |
| 适用规模 | 大规模 | 中小规模 |
| 条件数敏感性 | 敏感 | 不敏感 |

**选择建议**：
- 大规模问题（$n > 10^4$）：一阶方法或 L-BFGS
- 中小规模问题（$n < 10^4$）：牛顿法或拟牛顿法
- 精度要求高：二阶方法
- 实时应用：一阶方法

### 4.3.3 条件数与收敛速度

定义 4.4（条件数） 对于凸二次函数 $f(x) = \frac{1}{2}x^T Q x$，条件数定义为：
$$\kappa(Q) = \frac{\lambda_{\max}(Q)}{\lambda_{\min}(Q)}$$

定理 4.6 梯度下降法的收敛速度与条件数密切相关。对于强凸二次函数：
$$\|x_k - x^*\|_2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|x_0 - x^*\|_2$$

条件数越大，收敛越慢。

**预处理技术**

通过变量替换 $x = Py$ 改善条件数，其中 $P$ 是预处理矩阵。

---

## 4.4 本章小结

本章系统介绍了求解凸优化问题的主要算法：

1. **一阶方法**：
   - 梯度下降法：简单，收敛速度 $O(1/k)$
   - 随机梯度下降：适用于大规模问题
   - 投影梯度下降：处理简单约束

2. **二阶方法**：
   - 牛顿法：二次收敛，计算代价高
   - 拟牛顿法（BFGS、L-BFGS）：超线性收敛，折中方案
   - 共轭梯度法：适用于大规模问题

3. **约束优化算法**：
   - 投影梯度法：简单约束
   - 内点法：一般不等式约束，多项式复杂度
   - 增广拉格朗日法：等式约束

4. **算法选择**：
   - 考虑问题规模、精度要求、条件数
   - 大规模问题优先一阶方法
   - 高精度要求考虑二阶方法

---

## 习题

### 基础题

4.1 证明：对于 $L$-光滑凸函数，梯度下降法使用固定步长 $\alpha = 1/L$ 时，有：
$$f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|_2^2}{2k}$$

4.2 实现带回溯线搜索的梯度下降法，并在 Rosenbrock 函数上测试。

4.3 比较牛顿法和 BFGS 在求解二次函数时的迭代次数。

4.4 推导非负象限上的投影算子。

### 提高题

4.5 证明 BFGS 更新保持正定性（若 $H_k \succ 0$ 且 $y_k^T s_k > 0$，则 $H_{k+1} \succ 0$）。

4.6 分析 SGD 的方差对收敛的影响，并说明小批量 SGD 如何改善这一问题。

4.7 对于逻辑回归问题，推导 Hessian 矩阵并分析其条件数。

4.8 设计一个混合算法：前期使用梯度下降，后期切换到牛顿法。

### 编程题

4.9 实现 L-BFGS 算法，并与 scipy.optimize 的实现比较。

4.10 实现原对偶内点法求解 LP，测试不同规模的问题。

---

## 参考文献

1. Nocedal J, Wright S. Numerical Optimization[M]. Springer, 2006.
2. Boyd S, Vandenberghe L. Convex Optimization[M]. Cambridge University Press, 2004.
3. Bottou L, Curtis F E, Nocedal J. Optimization Methods for Large-Scale Machine Learning[J]. SIAM Review, 2018.
4. Wright S J. Coordinate Descent Algorithms[J]. Mathematical Programming, 2015.
