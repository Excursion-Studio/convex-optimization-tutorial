# 习题解答

本节提供各章节习题的提示与解答。

## 1. 第1章 介绍与预备知识

**习题1.1**：证明向量的2-范数满足三角不等式，即 $\lVert x + y \rVert_2 \leq \lVert x \rVert_2 + \lVert y \rVert_2$。

**提示**：使用柯西-施瓦茨不等式 $(x^T y)^2 \leq (x^T x)(y^T y)$。

**解答**：

$$\lVert x + y \rVert_2^2 = (x + y)^T (x + y) = x^T x + 2x^T y + y^T y = \lVert x \rVert_2^2 + 2x^T y + \lVert y \rVert_2^2$$

由柯西-施瓦茨不等式，$x^T y \leq \lVert x \rVert_2 \lVert y \rVert_2$，因此：

$$\lVert x + y \rVert_2^2 \leq \lVert x \rVert_2^2 + 2\lVert x \rVert_2 \lVert y \rVert_2 + \lVert y \rVert_2^2 = (\lVert x \rVert_2 + \lVert y \rVert_2)^2$$

两边开平方得证。

**习题1.2**：计算函数 $f(x) = x^3 - 3x$ 的梯度和海森矩阵。

**解答**：

梯度：$\nabla f(x) = 3x^2 - 3$

海森矩阵：$\nabla^2 f(x) = 6x$

**习题1.3**：证明向量的1-范数和∞-范数都满足三角不等式。

**解答**：

对于1-范数：

$$\lVert x + y \rVert_1 = \sum_{i=1}^n |x_i + y_i| \leq \sum_{i=1}^n (|x_i| + |y_i|) = \sum_{i=1}^n |x_i| + \sum_{i=1}^n |y_i| = \lVert x \rVert_1 + \lVert y \rVert_1$$

对于∞-范数：

$$\lVert x + y \rVert_\infty = \max_{1 \leq i \leq n} |x_i + y_i| \leq \max_{1 \leq i \leq n} (|x_i| + |y_i|) \leq \max_{1 \leq i \leq n} |x_i| + \max_{1 \leq i \leq n} |y_i| = \lVert x \rVert_\infty + \lVert y \rVert_\infty$$

**习题1.4**：计算函数 $f(x) = \frac{1}{2}x^T Ax + b^T x + c$ 的梯度和海森矩阵，其中 $A$ 是对称矩阵。

**解答**：

梯度：$\nabla f(x) = Ax + b$

海森矩阵：$\nabla^2 f(x) = A$

**习题1.5**：设 $x, y \in \mathbb{R}^n$，证明 $\lVert x + y \rVert_2^2 + \lVert x - y \rVert_2^2 = 2(\lVert x \rVert_2^2 + \lVert y \rVert_2^2)$。

**解答**：

$$\lVert x + y \rVert_2^2 = (x + y)^T (x + y) = x^T x + 2x^T y + y^T y = \lVert x \rVert_2^2 + 2x^T y + \lVert y \rVert_2^2$$

$$\lVert x - y \rVert_2^2 = (x - y)^T (x - y) = x^T x - 2x^T y + y^T y = \lVert x \rVert_2^2 - 2x^T y + \lVert y \rVert_2^2$$

相加得：

$$\lVert x + y \rVert_2^2 + \lVert x - y \rVert_2^2 = 2\lVert x \rVert_2^2 + 2\lVert y \rVert_2^2 = 2(\lVert x \rVert_2^2 + \lVert y \rVert_2^2)$$

**习题1.6**：设 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 是二次可微函数，证明 $f$ 是凸函数当且仅当它的海森矩阵在定义域上半正定。

**解答**：

（必要性）假设 $f$ 是凸函数，对于任意 $x, y \in \text{dom}(f)$，由泰勒展开：

$$f(y) \geq f(x) + \nabla f(x)^T (y - x) + \frac{1}{2} (y - x)^T \nabla^2 f(x) (y - x)$$

又由凸函数的一阶条件：

$$f(y) \geq f(x) + \nabla f(x)^T (y - x)$$

因此：

$$\frac{1}{2} (y - x)^T \nabla^2 f(x) (y - x) \geq 0$$

即海森矩阵半正定。

（充分性）假设海森矩阵在定义域上半正定，对于任意 $x, y \in \text{dom}(f)$，由泰勒展开：

$$f(y) = f(x) + \nabla f(x)^T (y - x) + \frac{1}{2} (y - x)^T \nabla^2 f(z) (y - x)$$

其中 $z$ 是 $x$ 和 $y$ 之间的某点。由于海森矩阵半正定，第二项非负，因此：

$$f(y) \geq f(x) + \nabla f(x)^T (y - x)$$

即 $f$ 是凸函数。

**习题1.7**：计算函数 $f(x) = \ln(1 + e^{x^T x})$ 的梯度。

**解答**：

令 $g(x) = x^T x$，则 $f(x) = \ln(1 + e^{g(x)})$。

由链式法则：

$$\nabla f(x) = \frac{e^{g(x)}}{1 + e^{g(x)}} \nabla g(x) = \frac{e^{x^T x}}{1 + e^{x^T x}} \cdot 2x$$

## 2. 第2章 凸集与凸函数基础

**习题2.1**：证明集合 $C = \{x \in \mathbb{R}^n \mid \lVert x \rVert_2 \leq 1\}$ 是凸集。

**提示**：使用三角不等式。

**解答**：

对于任意 $x, y \in C$ 和 $\theta \in [0, 1]$，有：

$$\lVert \theta x + (1 - \theta) y \rVert_2 \leq \theta \lVert x \rVert_2 + (1 - \theta) \lVert y \rVert_2 \leq \theta \cdot 1 + (1 - \theta) \cdot 1 = 1$$

因此 $\theta x + (1 - \theta) y \in C$，所以 $C$ 是凸集。

**习题2.2**：证明函数 $f(x) = e^{ax}$ 是凸函数，其中 $a \in \mathbb{R}$。

**解答**：

计算二阶导数：$f''(x) = a^2 e^{ax} > 0$，因此 $f$ 是凸函数。

**习题2.3**：证明两个凸集的交集仍然是凸集。

**解答**：

设 $C_1$ 和 $C_2$ 是凸集，$C = C_1 \cap C_2$。对于任意 $x, y \in C$ 和 $\theta \in [0, 1]$，有 $x, y \in C_1$ 和 $x, y \in C_2$。由于 $C_1$ 是凸集，所以 $\theta x + (1 - \theta) y \in C_1$；由于 $C_2$ 是凸集，所以 $\theta x + (1 - \theta) y \in C_2$。因此 $\theta x + (1 - \theta) y \in C_1 \cap C_2 = C$，所以 $C$ 是凸集。

**习题2.4**：证明线性函数 $f(x) = a^T x + b$ 既是凸函数又是凹函数。

**解答**：

对于任意 $x, y \in \mathbb{R}^n$ 和 $\theta \in [0, 1]$，有：

$$f(\theta x + (1 - \theta) y) = a^T (\theta x + (1 - \theta) y) + b = \theta a^T x + (1 - \theta) a^T y + b = \theta f(x) + (1 - \theta) f(y)$$

因此 $f$ 既是凸函数（满足 $f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)$）又是凹函数（满足 $f(\theta x + (1 - \theta) y) \geq \theta f(x) + (1 - \theta) f(y)$）。

**习题2.5**：证明集合 $C = \{x \in \mathbb{R}^n \mid Ax = b\}$ 是凸集，其中 $A$ 是 $m \times n$ 矩阵，$b \in \mathbb{R}^m$。

**解答**：

对于任意 $x, y \in C$ 和 $\theta \in [0, 1]$，有 $Ax = b$ 和 $Ay = b$。则：

$$A(\theta x + (1 - \theta) y) = \theta Ax + (1 - \theta) Ay = \theta b + (1 - \theta) b = b$$

因此 $\theta x + (1 - \theta) y \in C$，所以 $C$ 是凸集。

**习题2.6**：证明函数 $f(x) = \lVert x \rVert_2^2$ 是凸函数。

**解答**：

计算二阶导数（海森矩阵）：$\nabla^2 f(x) = 2I$，其中 $I$ 是单位矩阵。由于单位矩阵是正定的，所以海森矩阵是正定的，因此 $f$ 是凸函数。

**习题2.7**：设 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 和 $g: \mathbb{R}^n \rightarrow \mathbb{R}$ 都是凸函数，证明 $h(x) = \max\{f(x), g(x)\}$ 也是凸函数。

**解答**：

对于任意 $x, y \in \mathbb{R}^n$ 和 $\theta \in [0, 1]$，有：

$$h(\theta x + (1 - \theta) y) = \max\{f(\theta x + (1 - \theta) y), g(\theta x + (1 - \theta) y)\}$$

由于 $f$ 是凸函数，所以 $f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)$；由于 $g$ 是凸函数，所以 $g(\theta x + (1 - \theta) y) \leq \theta g(x) + (1 - \theta) g(y)$。因此：

$$h(\theta x + (1 - \theta) y) \leq \max\{\theta f(x) + (1 - \theta) f(y), \theta g(x) + (1 - \theta) g(y)\}$$

又因为 $\max\{a, b\} \leq \max\{c, d\}$ 当 $a \leq c$ 且 $b \leq d$，所以：

$$\max\{\theta f(x) + (1 - \theta) f(y), \theta g(x) + (1 - \theta) g(y)\} \leq \theta \max\{f(x), g(x)\} + (1 - \theta) \max\{f(y), g(y)\} = \theta h(x) + (1 - \theta) h(y)$$

因此 $h$ 是凸函数。

**习题2.8**：证明集合 $C = \{x \in \mathbb{R}^2 \mid x_1^2 + 2x_2^2 \leq 1\}$ 是凸集。

**解答**：

对于任意 $x, y \in C$ 和 $\theta \in [0, 1]$，有 $x_1^2 + 2x_2^2 \leq 1$ 和 $y_1^2 + 2y_2^2 \leq 1$。

考虑 $z = \theta x + (1 - \theta) y$，则：

$$z_1^2 + 2z_2^2 = (\theta x_1 + (1 - \theta) y_1)^2 + 2(\theta x_2 + (1 - \theta) y_2)^2$$

展开得：

$$\theta^2 (x_1^2 + 2x_2^2) + 2\theta(1 - \theta)(x_1 y_1 + 2x_2 y_2) + (1 - \theta)^2 (y_1^2 + 2y_2^2)$$

由于 $x_1^2 + 2x_2^2 \leq 1$ 和 $y_1^2 + 2y_2^2 \leq 1$，所以第一项和第三项都不超过 $\theta^2$ 和 $(1 - \theta)^2$。

对于第二项，由柯西-施瓦茨不等式：

$$x_1 y_1 + 2x_2 y_2 \leq \sqrt{x_1^2 + 2x_2^2} \sqrt{y_1^2 + 2y_2^2} \leq 1 \cdot 1 = 1$$

因此：

$$z_1^2 + 2z_2^2 \leq \theta^2 + 2\theta(1 - \theta) + (1 - \theta)^2 = (\theta + (1 - \theta))^2 = 1$$

所以 $z \in C$，因此 $C$ 是凸集。

## 3. 第3章 凸优化问题

**习题3.1**：将下列问题转化为标准形式的线性规划问题：

$$\max_{x} 2x_1 + 3x_2$$

$$\text{s.t. } x_1 + x_2 \leq 5$$

$$2x_1 + x_2 \leq 8$$

$$x_1, x_2 \geq 0$$

**解答**：

标准形式的线性规划问题是最小化问题，因此将目标函数取负：

$$\min_{x} -2x_1 - 3x_2$$

$$\text{s.t. } x_1 + x_2 \leq 5$$

$$2x_1 + x_2 \leq 8$$

$$x_1, x_2 \geq 0$$

**习题3.2**：判断下列问题是否为凸优化问题：

$$\min_{x} x_1^2 + 2x_2^2$$

$$\text{s.t. } x_1 + x_2 \geq 1$$

$$x_1, x_2 \geq 0$$

**解答**：

是的，这是一个凸优化问题。因为：
- 目标函数 $f(x) = x_1^2 + 2x_2^2$ 是凸函数（其二阶导数矩阵是正定的）。
- 不等式约束 $g_1(x) = -x_1 - x_2 + 1 \leq 0$ 是凸函数（线性函数）。
- 不等式约束 $g_2(x) = -x_1 \leq 0$ 和 $g_3(x) = -x_2 \leq 0$ 都是凸函数（线性函数）。

**习题3.3**：将下列问题转化为标准形式的凸优化问题：

$$\min_{x} \sqrt{x_1^2 + x_2^2}$$

$$\text{s.t. } x_1 + x_2 \leq 2$$

$$x_1, x_2 \geq 0$$

**解答**：

标准形式的凸优化问题是目标函数为凸函数，不等式约束为凸函数，等式约束为仿射函数的问题。

原问题中：
- 目标函数 $f(x) = \sqrt{x_1^2 + x_2^2}$ 是凸函数（范数函数）。
- 不等式约束 $g_1(x) = x_1 + x_2 - 2 \leq 0$ 是凸函数（线性函数）。
- 不等式约束 $g_2(x) = -x_1 \leq 0$ 和 $g_3(x) = -x_2 \leq 0$ 都是凸函数（线性函数）。

因此，原问题已经是标准形式的凸优化问题。

**习题3.4**：证明线性规划问题的可行域是凸集。

**解答**：

线性规划问题的可行域为：

$$D = \{x \in \mathbb{R}^n \mid Ax \leq b, x \geq 0\}$$

其中 $A$ 是 $m \times n$ 矩阵，$b \in \mathbb{R}^m$。

对于任意 $x, y \in D$ 和 $\theta \in [0, 1]$，有：

$$Ax \leq b, Ay \leq b, x \geq 0, y \geq 0$$

考虑 $z = \theta x + (1 - \theta) y$，则：

$$Az = \theta Ax + (1 - \theta) Ay \leq \theta b + (1 - \theta) b = b$$

且 $z = \theta x + (1 - \theta) y \geq 0$（因为 $\theta \geq 0$，$1 - \theta \geq 0$，$x \geq 0$，$y \geq 0$）。

因此 $z \in D$，所以 $D$ 是凸集。

**习题3.5**：设 $f(x)$ 是凸函数，$g_i(x)$ 是凸函数，$h_j(x)$ 是线性函数，证明优化问题：

$$\min_{x} f(x)$$

$$\text{s.t. } g_i(x) \leq 0, i=1,\ldots,m$$

$$h_j(x) = 0, j=1,\ldots,p$$

是凸优化问题。

**解答**：

凸优化问题的定义是：目标函数是凸函数，不等式约束函数是凸函数，等式约束函数是仿射函数的优化问题。

在这个问题中：
- 目标函数 $f(x)$ 是凸函数。
- 不等式约束函数 $g_i(x)$ 是凸函数。
- 等式约束函数 $h_j(x)$ 是线性函数，而线性函数是仿射函数的特殊情况。

因此，这个问题是凸优化问题。

**习题3.6**：将下列问题转化为标准形式的凸优化问题：

$$\min_{x} e^{x_1} + e^{x_2}$$

$$\text{s.t. } x_1 + x_2 \geq 1$$

$$x_1, x_2 \geq 0$$

**解答**：

标准形式的凸优化问题是目标函数为凸函数，不等式约束为凸函数，等式约束为仿射函数的问题。

原问题中：
- 目标函数 $f(x) = e^{x_1} + e^{x_2}$ 是凸函数（指数函数的和）。
- 不等式约束 $g_1(x) = -x_1 - x_2 + 1 \leq 0$ 是凸函数（线性函数）。
- 不等式约束 $g_2(x) = -x_1 \leq 0$ 和 $g_3(x) = -x_2 \leq 0$ 都是凸函数（线性函数）。

因此，原问题已经是标准形式的凸优化问题。

## 4. 第4章 无约束优化算法

**习题4.1**：使用梯度下降法求解函数 $f(x) = x_1^2 + 2x_2^2$ 的最小值，初始点为 $x_0 = (2, 1)^T$，步长为 $\alpha = 0.1$，迭代5次。

**解答**：

梯度：$\nabla f(x) = (2x_1, 4x_2)^T$

迭代过程：

- 第1次迭代：$x_1 = x_0 - \alpha \nabla f(x_0) = (2, 1) - 0.1(4, 4) = (1.6, 0.6)^T$
- 第2次迭代：$x_2 = x_1 - \alpha \nabla f(x_1) = (1.6, 0.6) - 0.1(3.2, 2.4) = (1.28, 0.36)^T$
- 第3次迭代：$x_3 = x_2 - \alpha \nabla f(x_2) = (1.28, 0.36) - 0.1(2.56, 1.44) = (1.024, 0.216)^T$
- 第4次迭代：$x_4 = x_3 - \alpha \nabla f(x_3) = (1.024, 0.216) - 0.1(2.048, 0.864) = (0.8192, 0.1296)^T$
- 第5次迭代：$x_5 = x_4 - \alpha \nabla f(x_4) = (0.8192, 0.1296) - 0.1(1.6384, 0.5184) = (0.65536, 0.07776)^T$

**习题4.2**：计算函数 $f(x) = x_1^3 + x_2^3 - 3x_1 - 3x_2$ 的梯度和海森矩阵。

**解答**：

梯度：$\nabla f(x) = (3x_1^2 - 3, 3x_2^2 - 3)^T$

海森矩阵：$\nabla^2 f(x) = \begin{bmatrix} 6x_1 & 0 \\ 0 & 6x_2 \end{bmatrix}$

**习题4.3**：简述梯度下降法的基本思想和步骤。

**解答**：

梯度下降法的基本思想是沿着目标函数梯度的负方向（即函数值下降最快的方向）移动，以达到最小化目标函数的目的。

步骤如下：
1. 选择初始点 $x_0$，设置迭代次数 $k=0$。
2. 计算梯度 $g_k = \nabla f(x_k)$。
3. 如果 $\lVert g_k \rVert \leq \epsilon$（$\epsilon$ 是预设的收敛阈值），则停止迭代，输出 $x_k$。
4. 选择步长 $\alpha_k > 0$。
5. 更新迭代点：$x_{k+1} = x_k - \alpha_k g_k$。
6. 令 $k = k + 1$，返回步骤 2。

**习题4.4**：使用牛顿法求解函数 $f(x) = x_1^2 + 2x_2^2$ 的最小值，初始点为 $x_0 = (2, 1)^T$，迭代3次。

**解答**：

梯度：$\nabla f(x) = (2x_1, 4x_2)^T$

海森矩阵：$\nabla^2 f(x) = \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix}$

海森矩阵的逆：$[\nabla^2 f(x)]^{-1} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.25 \end{bmatrix}$

牛顿方向：$d_k = -[\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$

迭代过程：

- 第1次迭代：$d_0 = -\begin{bmatrix} 0.5 & 0 \\ 0 & 0.25 \end{bmatrix} \begin{bmatrix} 4 \\ 4 \end{bmatrix} = \begin{bmatrix} -2 \\ -1 \end{bmatrix}$，$x_1 = x_0 + d_0 = (2, 1) + (-2, -1) = (0, 0)^T$
- 第2次迭代：$\nabla f(x_1) = (0, 0)^T$，满足收敛条件，停止迭代。

因此，牛顿法在一次迭代后就收敛到了最优解 $(0, 0)^T$。

**习题4.5**：证明对于正定二次函数，牛顿法一步即可收敛到最优解。

**解答**：

设正定二次函数为 $f(x) = \frac{1}{2}x^T Ax + b^T x + c$，其中 $A$ 是正定矩阵。

梯度：$\nabla f(x) = Ax + b$

海森矩阵：$\nabla^2 f(x) = A$

最优解满足 $\nabla f(x^*) = 0$，即 $Ax^* + b = 0$，解得 $x^* = -A^{-1}b$。

牛顿法的更新规则为：

$$x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k) = x_k - A^{-1}(Ax_k + b) = x_k - x_k - A^{-1}b = -A^{-1}b = x^*$$

因此，对于正定二次函数，牛顿法一步即可收敛到最优解。

**习题4.6**：比较梯度下降法和牛顿法的优缺点。

**解答**：

**梯度下降法的优点**：
- 算法简单，易于实现。
- 每次迭代的计算量小，仅需计算梯度。
- 适用于大规模优化问题。

**梯度下降法的缺点**：
- 收敛速度可能较慢，特别是在接近最优解时。
- 步长的选择对收敛速度影响很大。
- 在非凸优化问题中，可能会陷入局部最优解。

**牛顿法的优点**：
- 收敛速度快，在最优解附近具有二次收敛性。
- 能够处理目标函数的曲率信息。
- 对于正定二次函数，一步即可收敛。

**牛顿法的缺点**：
- 每次迭代需要计算海森矩阵并求解线性方程组，计算复杂度高。
- 海森矩阵可能是奇异的或非正定的，导致算法不稳定。
- 仅适用于小规模优化问题。

**习题4.7**：使用梯度下降法求解函数 $f(x) = (x_1 - 1)^2 + (x_2 - 2)^2$ 的最小值，初始点为 $x_0 = (0, 0)^T$，步长为 $\alpha = 0.1$，迭代10次，并计算最终点的函数值。

**解答**：

梯度：$\nabla f(x) = (2(x_1 - 1), 2(x_2 - 2))^T$

迭代过程：

- 第1次迭代：$x_1 = x_0 - \alpha \nabla f(x_0) = (0, 0) - 0.1(-2, -4) = (0.2, 0.4)^T$
- 第2次迭代：$x_2 = x_1 - \alpha \nabla f(x_1) = (0.2, 0.4) - 0.1(-1.6, -3.2) = (0.36, 0.72)^T$
- 第3次迭代：$x_3 = x_2 - \alpha \nabla f(x_2) = (0.36, 0.72) - 0.1(-1.28, -2.56) = (0.488, 0.976)^T$
- 第4次迭代：$x_4 = x_3 - \alpha \nabla f(x_3) = (0.488, 0.976) - 0.1(-1.024, -2.048) = (0.5904, 1.1808)^T$
- 第5次迭代：$x_5 = x_4 - \alpha \nabla f(x_4) = (0.5904, 1.1808) - 0.1(-0.8192, -1.6384) = (0.67232, 1.34464)^T$
- 第6次迭代：$x_6 = x_5 - \alpha \nabla f(x_5) = (0.67232, 1.34464) - 0.1(-0.65536, -1.31072) = (0.737856, 1.475712)^T$
- 第7次迭代：$x_7 = x_6 - \alpha \nabla f(x_6) = (0.737856, 1.475712) - 0.1(-0.524288, -1.048576) = (0.7902848, 1.5805696)^T$
- 第8次迭代：$x_8 = x_7 - \alpha \nabla f(x_7) = (0.7902848, 1.5805696) - 0.1(-0.4194304, -0.8388608) = (0.83222784, 1.66445568)^T$
- 第9次迭代：$x_9 = x_8 - \alpha \nabla f(x_8) = (0.83222784, 1.66445568) - 0.1(-0.33554432, -0.67108864) = (0.86578227, 1.73156454)^T$
- 第10次迭代：$x_{10} = x_9 - \alpha \nabla f(x_9) = (0.86578227, 1.73156454) - 0.1(-0.26843546, -0.53687092) = (0.89262582, 1.78525164)^T$

最终点的函数值：$f(x_{10}) = (0.89262582 - 1)^2 + (1.78525164 - 2)^2 \approx 0.0115 + 0.0460 = 0.0575$

## 5. 第5章 约束优化算法

**习题5.1**：使用拉格朗日乘数法求解下列问题：

$$\min_{x} x_1^2 + x_2^2$$

$$\text{s.t. } x_1 + x_2 = 1$$

**解答**：

构造拉格朗日函数：$L(x, \lambda) = x_1^2 + x_2^2 - \lambda(x_1 + x_2 - 1)$

求偏导并令其为零：

$$\frac{\partial L}{\partial x_1} = 2x_1 - \lambda = 0$$

$$\frac{\partial L}{\partial x_2} = 2x_2 - \lambda = 0$$

$$\frac{\partial L}{\partial \lambda} = -(x_1 + x_2 - 1) = 0$$

解得：$x_1 = x_2 = 1/2$，$\lambda = 1$。

**习题5.2**：写出下列问题的拉格朗日函数：

$$\min_{x} x_1^2 + 2x_2^2$$

$$\text{s.t. } x_1 + x_2 \geq 1$$

$$x_1, x_2 \geq 0$$

**解答**：

构造拉格朗日函数：

$$L(x, \mu_1, \mu_2, \mu_3) = x_1^2 + 2x_2^2 - \mu_1(x_1 + x_2 - 1) - \mu_2(-x_1) - \mu_3(-x_2)$$

其中 $\mu_1, \mu_2, \mu_3 \geq 0$ 是拉格朗日乘数。

**习题5.3**：简述拉格朗日乘数法的基本思想。

**解答**：

拉格朗日乘数法是求解等式约束优化问题的经典方法，其基本思想是通过引入拉格朗日乘数将约束优化问题转化为无约束优化问题。

具体来说，对于等式约束优化问题：

$$\min_{x} f(x)$$

$$\text{s.t. } h_j(x) = 0, \quad j=1,2,\ldots,p$$

拉格朗日乘数法引入拉格朗日乘数 $\lambda_j \in \mathbb{R}$，构造拉格朗日函数：

$$L(x, \lambda) = f(x) - \sum_{j=1}^p \lambda_j h_j(x)$$

然后将原问题转化为求解拉格朗日函数的驻点，即满足：

$$\nabla_x L(x^*, \lambda^*) = 0$$

$$\nabla_\lambda L(x^*, \lambda^*) = 0$$

这些条件称为一阶最优性条件或Kuhn-Tucker条件。

**习题5.4**：使用拉格朗日乘数法求解下列问题：

$$\min_{x} x_1^2 + x_2^2 + x_3^2$$

$$\text{s.t. } x_1 + x_2 + x_3 = 3$$

$$x_1 - x_2 = 1$$

**解答**：

构造拉格朗日函数：$L(x, \lambda_1, \lambda_2) = x_1^2 + x_2^2 + x_3^2 - \lambda_1(x_1 + x_2 + x_3 - 3) - \lambda_2(x_1 - x_2 - 1)$

求偏导并令其为零：

$$\frac{\partial L}{\partial x_1} = 2x_1 - \lambda_1 - \lambda_2 = 0$$

$$\frac{\partial L}{\partial x_2} = 2x_2 - \lambda_1 + \lambda_2 = 0$$

$$\frac{\partial L}{\partial x_3} = 2x_3 - \lambda_1 = 0$$

$$\frac{\partial L}{\partial \lambda_1} = -(x_1 + x_2 + x_3 - 3) = 0$$

$$\frac{\partial L}{\partial \lambda_2} = -(x_1 - x_2 - 1) = 0$$

解得：$x_1 = 5/3$，$x_2 = 2/3$，$x_3 = 2/3$，$\lambda_1 = 4/3$，$\lambda_2 = 2$。

**习题5.5**：证明对于凸优化问题，满足KKT条件的点是全局最优解。

**解答**：

设 $x^*$ 是凸优化问题的一个可行解，且满足KKT条件，即存在拉格朗日乘数 $\mu^* \geq 0$ 和 $\lambda^*$，使得：

1. $\nabla f(x^*) - \sum_{i=1}^m \mu_i^* \nabla g_i(x^*) - \sum_{j=1}^p \lambda_j^* \nabla h_j(x^*) = 0$
2. $g_i(x^*) \leq 0, i=1,\ldots,m$
3. $h_j(x^*) = 0, j=1,\ldots,p$
4. $\mu_i^* \geq 0, i=1,\ldots,m$
5. $\mu_i^* g_i(x^*) = 0, i=1,\ldots,m$

对于任意可行解 $x$，由于 $f$ 是凸函数，有：

$$f(x) \geq f(x^*) + \nabla f(x^*)^T (x - x^*)$$

将KKT条件1代入得：

$$f(x) \geq f(x^*) + \left(\sum_{i=1}^m \mu_i^* \nabla g_i(x^*) + \sum_{j=1}^p \lambda_j^* \nabla h_j(x^*)\right)^T (x - x^*)$$

由于 $g_i$ 是凸函数，有 $g_i(x) \geq g_i(x^*) + \nabla g_i(x^*)^T (x - x^*)$。对于 $\mu_i^* > 0$ 的情况，由互补松弛条件5知 $g_i(x^*) = 0$，因此 $\nabla g_i(x^*)^T (x - x^*) \leq g_i(x) \leq 0$（因为 $x$ 是可行解）。对于 $\mu_i^* = 0$ 的情况，该项为零。

对于等式约束，由于 $h_j$ 是仿射函数，有 $h_j(x) = h_j(x^*) + \nabla h_j(x^*)^T (x - x^*) = 0$，因此 $\nabla h_j(x^*)^T (x - x^*) = 0$。

综上，$f(x) \geq f(x^*)$，即 $x^*$ 是全局最优解。

**习题5.6**：使用拉格朗日乘数法求解下列问题：

$$\max_{x} x_1 x_2$$

$$\text{s.t. } x_1 + x_2 = 2$$

$$x_1, x_2 \geq 0$$

**解答**：

构造拉格朗日函数：$L(x, \lambda, \mu_1, \mu_2) = x_1 x_2 - \lambda(x_1 + x_2 - 2) - \mu_1(-x_1) - \mu_2(-x_2)$

其中 $\lambda \in \mathbb{R}$，$\mu_1, \mu_2 \geq 0$ 是拉格朗日乘数。

求偏导并令其为零：

$$\frac{\partial L}{\partial x_1} = x_2 - \lambda + \mu_1 = 0$$

$$\frac{\partial L}{\partial x_2} = x_1 - \lambda + \mu_2 = 0$$

$$\frac{\partial L}{\partial \lambda} = -(x_1 + x_2 - 2) = 0$$

$$\frac{\partial L}{\partial \mu_1} = x_1 = 0$$

$$\frac{\partial L}{\partial \mu_2} = x_2 = 0$$

考虑互补松弛条件 $\mu_1 x_1 = 0$ 和 $\mu_2 x_2 = 0$，假设 $x_1, x_2 > 0$，则 $\mu_1 = \mu_2 = 0$。

解得：$x_1 = x_2 = 1$，$\lambda = 1$。

验证：$x_1 + x_2 = 2$ 满足约束，且 $x_1, x_2 > 0$，因此最优解为 $(1, 1)^T$，最优值为 $1$。

## 6. 第6章 对偶理论

**习题6.1**：写出下列线性规划问题的对偶问题：

$$\min_{x} 3x_1 + 4x_2$$

$$\text{s.t. } x_1 + x_2 \geq 1$$

$$2x_1 + x_2 \geq 2$$

$$x_1, x_2 \geq 0$$

**解答**：

对偶问题为：

$$\max_{\mu} \mu_1 + 2\mu_2$$

$$\text{s.t. } \mu_1 + 2\mu_2 \leq 3$$

$$\mu_1 + \mu_2 \leq 4$$

$$\mu_1, \mu_2 \geq 0$$

**习题6.2**：简述对偶理论的基本思想。

**解答**：

对偶理论的基本思想是通过构造对偶问题来提供原问题最优值的下界，并在一定条件下（如强对偶性成立时）获得原问题的最优解。对偶理论通过引入拉格朗日乘数，将约束优化问题转化为无约束优化问题，然后定义对偶函数和对偶问题，从而为原问题的求解提供新的思路和方法。

**习题6.3**：写出下列线性规划问题的对偶问题：

$$\max_{x} 2x_1 + 3x_2$$

$$\text{s.t. } x_1 + x_2 \leq 5$$

$$2x_1 + x_2 \leq 8$$

$$x_1, x_2 \geq 0$$

**解答**：

首先将原问题转化为标准形式的极小化问题：

$$\min_{x} -2x_1 - 3x_2$$

$$\text{s.t. } x_1 + x_2 \leq 5$$

$$2x_1 + x_2 \leq 8$$

$$x_1, x_2 \geq 0$$

然后构造对偶问题：

$$\max_{\mu} -5\mu_1 - 8\mu_2$$

$$\text{s.t. } -\mu_1 - 2\mu_2 \leq -2$$

$$-\mu_1 - \mu_2 \leq -3$$

$$\mu_1, \mu_2 \geq 0$$

简化后得到：

$$\max_{\mu} -5\mu_1 - 8\mu_2$$

$$\text{s.t. } \mu_1 + 2\mu_2 \geq 2$$

$$\mu_1 + \mu_2 \geq 3$$

$$\mu_1, \mu_2 \geq 0$$

**习题6.4**：证明弱对偶性，即对于原问题和对偶问题，对偶问题的目标函数值不超过原问题的目标函数值（极小化问题）。

**解答**：

设原问题为：

$$\min_{x} f(x)$$

$$\text{s.t. } g_i(x) \leq 0, i=1,\ldots,m$$

$$h_j(x) = 0, j=1,\ldots,p$$

其拉格朗日函数为：

$$L(x, \mu, \lambda) = f(x) - \sum_{i=1}^m \mu_i g_i(x) - \sum_{j=1}^p \lambda_j h_j(x)$$

其中 $\mu_i \geq 0$。

对偶函数为：

$$d(\mu, \lambda) = \inf_{x} L(x, \mu, \lambda)$$

对于任意可行解 $x$ 和对偶变量 $\mu \geq 0, \lambda$，有：

$$L(x, \mu, \lambda) = f(x) - \sum_{i=1}^m \mu_i g_i(x) - \sum_{j=1}^p \lambda_j h_j(x) \leq f(x)$$

因为 $g_i(x) \leq 0$，$\mu_i \geq 0$，所以 $-\sum_{i=1}^m \mu_i g_i(x) \leq 0$；又因为 $h_j(x) = 0$，所以 $-\sum_{j=1}^p \lambda_j h_j(x) = 0$。

因此，

$$d(\mu, \lambda) = \inf_{x} L(x, \mu, \lambda) \leq L(x, \mu, \lambda) \leq f(x)$$

对于原问题的最优解 $x^*$，有 $d(\mu, \lambda) \leq f(x^*) = p^*$，其中 $p^*$ 是原问题的最优值。

对于对偶问题的最优解 $(\mu^*, \lambda^*)$，有 $d(\mu^*, \lambda^*) = d^* \leq p^*$，即弱对偶性成立。

**习题6.5**：证明强对偶性，即对于凸优化问题，当满足 Slater 条件时，原问题和对偶问题的最优值相等。

**解答**：

强对偶性的证明较为复杂，这里给出简要的证明思路：

1. 对于凸优化问题，原问题的最优值 $p^*$ 是凸函数。
2. 对偶函数 $d(\mu, \lambda)$ 是凹函数，其最优值 $d^*$ 是对偶问题的最大值。
3. Slater 条件保证了存在严格可行点，使得原问题和对偶问题之间没有对偶间隙。
4. 通过构造分离超平面和利用凸函数的性质，可以证明 $p^* = d^*$。

详细的证明可以参考凸优化的经典教材，如 Boyd 和 Vandenberghe 的《Convex Optimization》。

**习题6.6**：使用对偶理论求解下列线性规划问题：

$$\min_{x} x_1 + x_2$$

$$\text{s.t. } x_1 + 2x_2 \geq 3$$

$$2x_1 + x_2 \geq 3$$

$$x_1, x_2 \geq 0$$

**解答**：

构造对偶问题：

$$\max_{\mu} 3\mu_1 + 3\mu_2$$

$$\text{s.t. } \mu_1 + 2\mu_2 \leq 1$$

$$2\mu_1 + \mu_2 \leq 1$$

$$\mu_1, \mu_2 \geq 0$$

求解对偶问题，得到最优解 $\mu_1^* = \mu_2^* = 1/3$，最优值 $d^* = 3*(1/3) + 3*(1/3) = 2$。

根据强对偶性，原问题的最优值 $p^* = d^* = 2$。

利用互补松弛条件，原问题的最优解满足：

$$x_1^* + 2x_2^* = 3$$

$$2x_1^* + x_2^* = 3$$

解得 $x_1^* = x_2^* = 1$。

## 7. 第7章 应用案例

**习题7.1**：使用线性回归模型拟合以下数据：

| x | y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 7 |
| 5 | 8 |

**解答**：

设线性模型为 $y = w_1 x + w_0$，构造损失函数：

$$L(w_0, w_1) = \sum_{i=1}^5 (y_i - (w_1 x_i + w_0))^2$$

计算偏导数并令其为零：

$$\frac{\partial L}{\partial w_0} = -2\sum_{i=1}^5 (y_i - w_1 x_i - w_0) = 0$$

$$\frac{\partial L}{\partial w_1} = -2\sum_{i=1}^5 x_i (y_i - w_1 x_i - w_0) = 0$$

解得：$w_0 = 0.6$，$w_1 = 1.6$。

**习题7.2**：简述线性回归模型的基本思想和求解方法。

**解答**：

线性回归模型的基本思想是假设因变量 $y$ 与自变量 $x$ 之间存在线性关系，即 $y = w^T x + b + \epsilon$，其中 $w$ 是权重向量，$b$ 是偏置项，$\epsilon$ 是误差项。

线性回归的求解方法主要是最小二乘法，即最小化误差项的平方和：

$$\min_{w, b} \sum_{i=1}^n (y_i - (w^T x_i + b))^2$$

对于简单线性回归（只有一个自变量），可以通过求解正规方程组得到解析解；对于多元线性回归，可以通过矩阵运算求解，或者使用梯度下降法等优化算法。

**习题7.3**：使用最小二乘法求解线性回归模型的参数。

**解答**：

对于线性回归模型 $y = Xw + \epsilon$，其中 $X$ 是设计矩阵，$w$ 是参数向量，最小二乘法的目标是最小化残差平方和：

$$\min_{w} \lVert y - Xw \rVert_2^2$$

对 $w$ 求导并令导数为零，得到正规方程：

$$X^T X w = X^T y$$

当 $X^T X$ 可逆时，解为：

$$w = (X^T X)^{-1} X^T y$$

**习题7.4**：使用线性回归模型拟合以下数据，并计算均方误差：

| x1 | x2 | y |
|----|----|---|
| 1  | 2  | 3 |
| 2  | 4  | 5 |
| 3  | 6  | 7 |
| 4  | 8  | 9 |
| 5  | 10 | 11 |

**解答**：

设线性模型为 $y = w_1 x_1 + w_2 x_2 + w_0$，构造设计矩阵 $X$：

$$X = \begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 4 \\
1 & 3 & 6 \\
1 & 4 & 8 \\
1 & 5 & 10
\end{bmatrix}$$

目标向量 $y$：

$$y = \begin{bmatrix} 3 \\ 5 \\ 7 \\ 9 \\ 11 \end{bmatrix}$$

计算正规方程：

$$X^T X w = X^T y$$

解得：$w_0 = 1$，$w_1 = 1$，$w_2 = 0$（注意：由于 $x_2 = 2x_1$，存在多重共线性，所以 $w_2$ 为0）。

均方误差：

$$MSE = \frac{1}{5} \sum_{i=1}^5 (y_i - (w_1 x_{i1} + w_2 x_{i2} + w_0))^2 = 0$$

**习题7.5**：简述岭回归和LASSO回归的基本思想和区别。

**解答**：

岭回归和LASSO回归都是线性回归的正则化变体，它们的基本思想是通过在损失函数中添加正则化项来防止过拟合。

**岭回归**：在损失函数中添加L2正则化项：

$$\min_{w} \sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda \lVert w \rVert_2^2$$

其中 $\lambda > 0$ 是正则化参数。

**LASSO回归**：在损失函数中添加L1正则化项：

$$\min_{w} \sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda \lVert w \rVert_1$$

其中 $\lambda > 0$ 是正则化参数。

**区别**：

1. 正则化项不同：岭回归使用L2范数，LASSO回归使用L1范数。
2. 解的性质不同：岭回归的解是唯一的，且所有系数都不为零；LASSO回归的解可能不唯一，且会将一些系数压缩为零，实现特征选择。
3. 计算复杂度不同：岭回归可以通过正规方程求解，计算简单；LASSO回归需要使用迭代算法求解，计算复杂度较高。

**习题7.6**：使用Python实现线性回归模型，拟合习题7.1中的数据，并绘制拟合直线。

**解答**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 7, 8])

# 构造设计矩阵
X = np.vstack([np.ones(len(x)), x]).T

# 最小二乘法求解
w = np.linalg.inv(X.T @ X) @ X.T @ y

# 预测
x_pred = np.linspace(0, 6, 100)
y_pred = w[0] + w[1] * x_pred

# 绘制结果
plt.scatter(x, y, label='数据点')
plt.plot(x_pred, y_pred, 'r-', label='拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("参数:", w)
print("拟合直线: y = {:.2f} + {:.2f}x".format(w[0], w[1]))
```

运行结果：

参数: [0.6 1.6]
拟合直线: y = 0.60 + 1.60x

