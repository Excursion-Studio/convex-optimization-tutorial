# 第9章 流形优化的基本理论

## 9.1 流形上的优化问题

### 9.1.1 流形优化问题的定义

流形优化是一类约束优化问题，其中约束集合具有流形结构。与欧几里得空间中的优化不同，流形优化需要利用微分几何的工具来处理约束。

**定义 9.1（流形优化问题）** 设 $M$ 是一个光滑流形，$f: M \to \mathbb{R}$ 是光滑函数。**流形优化问题**定义为：
$$\min_{x \in M} f(x)$$

**例 9.1（特征值问题）** 设 $A$ 是 $n \times n$ 对称矩阵，求最小特征值等价于：
$$\min_{x \in S^{n-1}} x^T A x$$

其中 $S^{n-1} = \{x \in \mathbb{R}^n : \|x\| = 1\}$ 是单位球面。

**例 9.2（低秩矩阵逼近）** 给定矩阵 $A \in \mathbb{R}^{m \times n}$，寻找秩不超过 $r$ 的最佳逼近：
$$\min_{X \in \mathcal{M}_r} \|A - X\|_F^2$$

其中 $\mathcal{M}_r = \{X \in \mathbb{R}^{m \times n} : \text{rank}(X) = r\}$ 是固定秩矩阵流形。

**例 9.3（姿态估计）** 给定测量值，估计机器人的旋转姿态：
$$\min_{R \in SO(3)} \sum_{i=1}^n \|R p_i - q_i\|^2$$

其中 $SO(3)$ 是旋转群。

### 9.1.2 流形优化与欧几里得优化的区别

| 特性 | 欧几里得优化 | 流形优化 |
|------|-------------|----------|
| 搜索空间 | 向量空间 $\mathbb{R}^n$ | 流形 $M$ |
| 迭代更新 | $x_{k+1} = x_k + \alpha_k d_k$ | 沿测地线或利用收缩映射 |
| 梯度 | 普通梯度 $\nabla f$ | 黎曼梯度 $\text{grad} f$ |
| 约束处理 | 惩罚函数或投影 | 内蕴处理，自动满足约束 |
| 收敛分析 | 标准技术 | 需要黎曼几何工具 |

### 9.1.3 常见优化流形

**正交群**：
$$O(n) = \{Q \in \mathbb{R}^{n \times n} : Q^T Q = I\}$$

**特殊正交群**：
$$SO(n) = \{Q \in O(n) : \det(Q) = 1\}$$

**Stiefel流形**：
$$\text{St}(n, p) = \{X \in \mathbb{R}^{n \times p} : X^T X = I_p\}$$

**Grassmann流形**：
$$\text{Gr}(n, p) = \{\text{span}(X) : X \in \text{St}(n, p)\}$$

**固定秩矩阵流形**：
$$\mathcal{M}_r = \{X \in \mathbb{R}^{m \times n} : \text{rank}(X) = r\}$$

**对称正定矩阵流形**：
$$\mathcal{S}_+^n = \{X \in \mathbb{R}^{n \times n} : X = X^T, X \succ 0\}$$

## 9.2 流形上的梯度和海森矩阵

### 9.2.1 黎曼梯度

**定义 9.2（黎曼梯度）** 设 $f: M \to \mathbb{R}$ 是光滑函数，$M$ 是装备了度量 $g$ 的黎曼流形。**黎曼梯度** $\text{grad} f(x) \in T_x M$ 是唯一满足以下条件的切向量：
$$g_x(\text{grad} f(x), v) = df_x(v), \quad \forall v \in T_x M$$

在坐标下，若 $g_{ij}$ 是度量张量，则：
$$(\text{grad} f)^i = g^{ij} \frac{\partial f}{\partial x^j}$$

**定理 9.1** 黎曼梯度是 $f$ 的最速上升方向：
$$\frac{\text{grad} f(x)}{\|\text{grad} f(x)\|} = \arg\max_{v \in T_x M, \|v\|=1} df_x(v)$$

**例 9.4（球面上的梯度）** 在 $S^{n-1}$ 上，对于 $f(x) = x^T A x$，欧几里得梯度为 $2Ax$。投影到切空间 $T_x S^{n-1} = \{v : x^T v = 0\}$，得到黎曼梯度：
$$\text{grad} f(x) = 2Ax - 2(x^T A x)x = 2(A - (x^T A x)I)x$$

### 9.2.2 黎曼海森矩阵

**定义 9.3（黎曼海森）** **黎曼海森**（或**协变海森**）是映射 $\text{Hess} f: \mathfrak{X}(M) \times \mathfrak{X}(M) \to C^\infty(M)$，定义为：
$$\text{Hess} f(X, Y) = g(\nabla_X \text{grad} f, Y)$$

其中 $\nabla$ 是列维-奇维塔联络。

**定理 9.2** 黎曼海森是对称的：
$$\text{Hess} f(X, Y) = \text{Hess} f(Y, X)$$

**例 9.5（欧几里得空间）** 在 $\mathbb{R}^n$ 上，黎曼海森退化为普通海森矩阵：
$$\text{Hess} f(X, Y) = X^T \nabla^2 f Y$$

### 9.2.3 流形上的泰勒展开

**定理 9.3（流形上的泰勒展开）** 设 $f: M \to \mathbb{R}$ 是光滑函数，$\gamma$ 是满足 $\gamma(0) = x$ 和 $\dot{\gamma}(0) = v$ 的测地线。则：
$$f(\gamma(t)) = f(x) + t \cdot df_x(v) + \frac{t^2}{2} \cdot \text{Hess} f(v, v) + O(t^3)$$

这推广了欧几里得空间中的泰勒展开：
$$f(x + tv) = f(x) + t \nabla f(x)^T v + \frac{t^2}{2} v^T \nabla^2 f(x) v + O(t^3)$$

## 9.3 流形优化的最优性条件

### 9.3.1 一阶最优性条件

**定义 9.4（临界点）** 点 $x^* \in M$ 称为 $f$ 的**临界点**，如果：
$$\text{grad} f(x^*) = 0$$

**定理 9.4（一阶必要条件）** 若 $x^*$ 是 $f$ 的局部极小点，则 $x^*$ 是临界点。

**证明**：若 $\text{grad} f(x^*) \neq 0$，则沿 $-\text{grad} f(x^*)$ 方向函数值下降，与局部极小矛盾。

### 9.3.2 二阶最优性条件

**定理 9.5（二阶必要条件）** 若 $x^*$ 是 $f$ 的局部极小点，则：
1. $\text{grad} f(x^*) = 0$
2. $\text{Hess} f(x^*) \succeq 0$（半正定）

**定理 9.6（二阶充分条件）** 若 $x^* \in M$ 满足：
1. $\text{grad} f(x^*) = 0$
2. $\text{Hess} f(x^*) \succ 0$（正定）

则 $x^*$ 是 $f$ 的严格局部极小点。

### 9.3.3 流形优化的对偶理论

对于带约束的流形优化问题，可以发展对偶理论。

**定义 9.5（拉格朗日函数）** 考虑问题：
$$\min_{x \in M} f(x) \quad \text{s.t.} \quad h(x) = 0$$

其中 $h: M \to \mathbb{R}^m$。**拉格朗日函数**为：
$$\mathcal{L}(x, \lambda) = f(x) + \lambda^T h(x)$$

**定理 9.7（KKT条件）** 在正则性条件下，若 $x^*$ 是局部最优解，则存在 $\lambda^* \in \mathbb{R}^m$ 使得：
1. $\text{grad}_x \mathcal{L}(x^*, \lambda^*) = 0$
2. $h(x^*) = 0$

## 9.4 收缩映射和向量传输

### 9.4.1 收缩映射

在流形优化中，需要一种方式将切向量映射回流形，这称为**收缩**。

**定义 9.6（收缩映射）** 映射 $R: TM \to M$ 称为**收缩映射**，如果：
1. $R_x(0_x) = x$，其中 $0_x$ 是 $T_x M$ 的零向量
2. $\left.\frac{d}{dt}\right|_{t=0} R_x(tv) = v$，即微分 $d(R_x)_0 = \text{id}_{T_x M}$

**例 9.6（球面上的收缩）** 在 $S^{n-1}$ 上，常用的收缩为：
$$R_x(v) = \frac{x + v}{\|x + v\|}$$

**例 9.7（矩阵流形上的收缩）** 在正交群 $O(n)$ 上，收缩可以是：
$$R_X(V) = \text{qf}(X + V)$$

其中 $\text{qf}$ 表示QR分解的Q因子。

### 9.4.2 向量传输

在优化迭代中，需要将不同点的切向量进行比较，这需要**向量传输**。

**定义 9.7（向量传输）** **向量传输** $\mathcal{T}: TM \oplus TM \to TM$ 是一个映射，将 $(x, v, w)$（其中 $v, w \in T_x M$）映射到 $T_{R_x(v)} M$ 中的向量。

常见的向量传输包括：
- **平行移动**：沿测地线平行移动
- **投影传输**：将向量投影到新的切空间

**例 9.8（球面上的传输）** 在 $S^{n-1}$ 上，从 $x$ 到 $y$ 的投影传输为：
$$\mathcal{T}_{x \to y}(v) = v - (y^T v)y$$

## 9.5 本章小结

本章介绍了流形优化的基本理论：

1. **流形优化问题**：在流形上最小化目标函数，约束自动满足
2. **黎曼梯度**：最速上升方向，依赖于度量结构
3. **黎曼海森**：二阶信息，用于设计牛顿类算法
4. **最优性条件**：一阶和二阶条件推广到流形
5. **收缩和传输**：流形优化的基本运算工具

这些理论为下一章介绍流形优化算法奠定了基础。

## 9.6 参考文献

1. Absil, P. A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
2. Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press.
3. Hu, J., Milzarek, A., Wen, Z., & Yuan, Y. (2019). Adaptive quadratically regularized Newton method for Riemannian optimization. *SIAM Journal on Matrix Analysis and Applications*, 40(3), 1189-1212.

## 9.7 练习题

1. 证明黎曼梯度 $\text{grad} f(x)$ 是最速上升方向。

2. 在 $SO(3)$ 上，设 $f(R) = \|R - A\|_F^2$，计算黎曼梯度。

3. 证明黎曼海森的对称性。

4. 验证球面上的收缩映射 $R_x(v) = \frac{x+v}{\|x+v\|}$ 满足收缩的定义。

5. 设 $f: M \to \mathbb{R}$ 是光滑函数，$\gamma$ 是测地线。证明：
   $$\frac{d^2}{dt^2}f(\gamma(t)) = \text{Hess} f(\dot{\gamma}(t), \dot{\gamma}(t))$$

6. 考虑Grassmann流形上的优化问题。描述其切空间和投影操作。
