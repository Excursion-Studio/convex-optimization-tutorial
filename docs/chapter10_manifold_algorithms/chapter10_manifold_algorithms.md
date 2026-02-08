# 第10章 流形优化算法

## 10.1 流形上的梯度下降法

### 10.1.1 黎曼梯度下降法

**算法 10.1（黎曼梯度下降法）**

输入：初始点 $x_0 \in M$，步长序列 $\{\alpha_k\}$

对于 $k = 0, 1, 2, \ldots$：
1. 计算黎曼梯度：$g_k = \text{grad} f(x_k)$
2. 选择搜索方向：$d_k = -g_k$
3. 沿测地线或收缩更新：
   $$x_{k+1} = R_{x_k}(\alpha_k d_k)$$

**步长选择**：
- **固定步长**：$\alpha_k = \alpha$
- **递减步长**：$\alpha_k = \frac{\alpha}{\sqrt{k+1}}$
- **线搜索**：Armijo条件、Wolfe条件

### 10.1.2 步长选择和线搜索

**Armijo条件**：
$$f(R_{x_k}(\alpha d_k)) \leq f(x_k) + c_1 \alpha \langle g_k, d_k \rangle_{x_k}$$

其中 $c_1 \in (0, 1)$。

**回溯线搜索**：
1. 初始化 $\alpha = \alpha_0$
2. 重复直到满足Armijo条件：
   $$\alpha \leftarrow \beta \alpha$$

其中 $\beta \in (0, 1)$。

### 10.1.3 收敛性分析

**定理 10.1（梯度下降的收敛性）** 设 $f$ 是Lipschitz连续可微的，步长满足一定条件，则：
$$\lim_{k \to \infty} \|\text{grad} f(x_k)\| = 0$$

**收敛速度**：
- 凸问题：$O(1/k)$
- 强凸问题：$O(\rho^k)$（线性收敛）

## 10.2 流形上的牛顿法

### 10.2.1 黎曼牛顿法的构造

**算法 10.2（黎曼牛顿法）**

对于 $k = 0, 1, 2, \ldots$：
1. 计算黎曼梯度和海森：$g_k = \text{grad} f(x_k)$，$H_k = \text{Hess} f(x_k)$
2. 求解牛顿方程：
   $$H_k[d_k] = -g_k$$
3. 更新：$x_{k+1} = R_{x_k}(d_k)$

### 10.2.2 海森矩阵的近似计算

**拟牛顿法**：避免直接计算海森。

**BFGS在流形上的推广**：
- 使用向量传输代替平移
- 更新近似海森

### 10.2.3 牛顿法的收敛速度

**定理 10.2（牛顿法的局部收敛性）** 在临界点 $x^*$ 附近，若海森正定，则牛顿法具有**二次收敛速度**：
$$\text{dist}(x_{k+1}, x^*) \leq C \cdot \text{dist}(x_k, x^*)^2$$

## 10.3 其他流形优化算法

### 10.3.1 流形上的共轭梯度法

**算法 10.3（黎曼共轭梯度法）**

初始化：$x_0$，$g_0 = \text{grad} f(x_0)$，$d_0 = -g_0$

对于 $k = 0, 1, 2, \ldots$：
1. 线搜索：$\alpha_k = \arg\min_\alpha f(R_{x_k}(\alpha d_k))$
2. 更新：$x_{k+1} = R_{x_k}(\alpha_k d_k)$
3. 计算新梯度：$g_{k+1} = \text{grad} f(x_{k+1})$
4. 传输旧梯度：$\tilde{g}_k = \mathcal{T}_{x_k \to x_{k+1}}(g_k)$
5. 计算参数（如Fletcher-Reeves）：
   $$\beta_{k+1} = \frac{\langle g_{k+1}, g_{k+1} \rangle_{x_{k+1}}}{\langle \tilde{g}_k, \tilde{g}_k \rangle_{x_{k+1}}}$$
6. 更新方向：$d_{k+1} = -g_{k+1} + \beta_{k+1} \mathcal{T}_{x_k \to x_{k+1}}(d_k)$

### 10.3.2 流形上的拟牛顿法

**黎曼BFGS**：
- 维护海森的近似 $B_k$
- 使用向量传输处理不同切空间的向量

**更新公式**：
$$B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}$$

其中 $s_k$ 和 $y_k$ 需要通过向量传输到同一空间。

### 10.3.3 随机流形优化算法

**随机梯度下降（Riemannian SGD）**：

对于大规模问题，使用随机梯度代替全梯度：
$$x_{k+1} = R_{x_k}(-\alpha_k \text{grad} f_{i_k}(x_k))$$

其中 $i_k$ 是随机采样的样本索引。

**方差缩减方法**：
- SVRG（Riemannian版本）
- SAGA（Riemannian版本）

## 10.4 本章小结

本章介绍了流形优化的主要算法：

1. **黎曼梯度下降**：基本的一阶算法
2. **黎曼牛顿法**：二阶算法，具有二次收敛速度
3. **黎曼共轭梯度**：结合梯度下降和牛顿法的优点
4. **黎曼拟牛顿法**：避免计算海森
5. **随机算法**：适用于大规模问题

这些算法将欧几里得空间中的经典优化算法推广到流形上，在机器人姿态估计、计算机视觉等领域有广泛应用。

## 10.5 参考文献

1. Absil, P. A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
2. Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press.
3. Zhang, H., & Sra, S. (2016). First-order methods for geodesically convex optimization. *Conference on Learning Theory*, 1617-1638.

## 10.6 练习题

1. 实现黎曼梯度下降法求解球面上的特征值问题。

2. 比较黎曼梯度下降和黎曼牛顿法的收敛速度。

3. 推导黎曼共轭梯度法的Fletcher-Reeves参数。

4. 解释为什么在流形优化中需要向量传输。

5. 实现黎曼BFGS算法并测试其性能。
