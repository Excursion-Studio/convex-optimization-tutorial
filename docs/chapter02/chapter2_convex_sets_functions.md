# 第2章 凸集与凸函数基础

## 2.1 凸集

### 2.1.1 凸集的定义与基本性质

#### 凸集的定义

在欧几里得空间 $\mathbb{R}^n$ 中，一个集合 $C$ 被称为凸集，如果对于任意的 $x, y \in C$ 和任意的 $\theta \in [0, 1]$，都有：

$$\theta x + (1 - \theta) y \in C$$

换句话说，凸集是这样一种集合，其中任意两点之间的线段完全包含在该集合中。

#### 凸集的基本性质

1. **空集**：空集 $\emptyset$ 是凸集。
2. **单点集**：任意单点集 $\{x\}$ 是凸集。
3. **全空间**：整个空间 $\mathbb{R}^n$ 是凸集。
4. **交集**：任意多个凸集的交集仍然是凸集。即，如果 $C_1, C_2, \ldots, C_k$ 都是凸集，那么 $\bigcap_{i=1}^k C_i$ 也是凸集。
5. **仿射组合**：如果 $C$ 是凸集，那么对于任意的 $x_1, x_2, \ldots, x_k \in C$ 和非负实数 $\theta_1, \theta_2, \ldots, \theta_k$ 满足 $\sum_{i=1}^k \theta_i = 1$，有 $\sum_{i=1}^k \theta_i x_i \in C$。

### 2.1.2 重要的凸集示例

#### 仿射集

仿射集是凸集的一种特殊情况。一个集合 $A$ 是仿射集，如果对于任意的 $x, y \in A$ 和任意的 $\theta \in \mathbb{R}$，都有：

$$\theta x + (1 - \theta) y \in A$$

仿射集的例子包括：
- 空集
- 单点集
- 直线
- 平面
- 子空间

#### 线性子空间

线性子空间是包含原点的仿射集。如果 $V$ 是 $\mathbb{R}^n$ 的一个子集，且满足：
1. 对于任意的 $x, y \in V$，有 $x + y \in V$；
2. 对于任意的 $x \in V$ 和 $\alpha \in \mathbb{R}$，有 $\alpha x \in V$。

则 $V$ 是一个线性子空间。

#### 多面体

多面体是由有限个线性不等式和等式定义的集合，即：

$$P = \{x \in \mathbb{R}^n \mid Ax \leq b, Cx = d\}$$

其中 $A \in \mathbb{R}^{m \times n}$，$b \in \mathbb{R}^m$，$C \in \mathbb{R}^{p \times n}$，$d \in \mathbb{R}^p$。

#### 半空间

半空间是由单个线性不等式定义的集合：

$$H = \{x \in \mathbb{R}^n \mid a^T x \leq b\}$$

其中 $a \in \mathbb{R}^n \setminus \{0\}$，$b \in \mathbb{R}$。半空间是凸集。

#### 超平面

超平面是由单个线性等式定义的集合：

$$H = \{x \in \mathbb{R}^n \mid a^T x = b\}$$

其中 $a \in \mathbb{R}^n \setminus \{0\}$，$b \in \mathbb{R}$。超平面是仿射集，因此也是凸集。

#### 球

欧几里得球（简称球）是指所有与给定点距离不超过某个半径的点的集合：

$$B(x_c, r) = \{x \in \mathbb{R}^n \mid \lVert x - x_c \rVert_2 \leq r\}$$

其中 $x_c \in \mathbb{R}^n$ 是球心，$r > 0$ 是半径。球是凸集。

#### 半正定锥

半正定锥是所有 $n \times n$ 半正定矩阵的集合：

$$S_+^n = \{X \in \mathbb{S}^n \mid X \succeq 0\}$$

其中 $\mathbb{S}^n$ 是所有 $n \times n$ 对称矩阵的集合。半正定锥是凸集。

### 2.1.3 凸集的运算

凸集在以下运算下保持凸性：

1. **交集**：任意多个凸集的交集仍然是凸集。
2. **仿射变换**：如果 $C$ 是凸集，$f(x) = Ax + b$ 是仿射变换（$A \in \mathbb{R}^{m \times n}$，$b \in \mathbb{R}^m$），则 $f(C) = \{Ax + b \mid x \in C\}$ 是凸集。
3. **逆仿射变换**：如果 $C$ 是凸集，$f(x) = Ax + b$ 是仿射变换，则 $f^{-1}(C) = \{x \mid Ax + b \in C\}$ 是凸集。
4. **透视函数**：透视函数 $P: \mathbb{R}^{n+1} \to \mathbb{R}^n$ 定义为 $P(x, t) = x/t$，其中 $t > 0$。如果 $C \subseteq \mathbb{R}^{n+1}$ 是凸集，则 $P(C) = \{x/t \mid (x, t) \in C, t > 0\}$ 是凸集。
5. **线性分式函数**：线性分式函数是透视函数和仿射变换的组合。如果 $C$ 是凸集，则线性分式函数作用于 $C$ 的结果仍然是凸集。

### 2.1.4 分离定理与支撑超平面

#### 超平面分离定理

**超平面分离定理**：如果 $C$ 和 $D$ 是两个不相交的凸集，即 $C \cap D = \emptyset$，则存在非零向量 $a \in \mathbb{R}^n$ 和标量 $b \in \mathbb{R}$，使得对于所有 $x \in C$，有 $a^T x \leq b$，对于所有 $x \in D$，有 $a^T x \geq b$。

换句话说，存在一个超平面将两个不相交的凸集分离开来。

#### 支撑超平面

设 $C \subseteq \mathbb{R}^n$ 是一个凸集，$x_0$ 是 $C$ 的边界点，即 $x_0 \in \text{bd}(C)$。如果存在非零向量 $a \in \mathbb{R}^n$，使得对于所有 $x \in C$，有 $a^T x \leq a^T x_0$，则称超平面 $\{x \mid a^T x = a^T x_0\}$ 是 $C$ 在点 $x_0$ 处的支撑超平面。

**支撑超平面定理**：如果 $C \subseteq \mathbb{R}^n$ 是一个凸集，$x_0$ 是 $C$ 的边界点，则 $C$ 在 $x_0$ 处存在支撑超平面。

## 2.2 凸函数

### 2.2.1 凸函数的定义与一阶、二阶条件

#### 凸函数的定义

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 被称为凸函数，如果其定义域 $\text{dom}(f)$ 是凸集，并且对于任意的 $x, y \in \text{dom}(f)$ 和任意的 $\theta \in [0, 1]$，都有：

$$f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)$$

如果上述不等式对于 $x \neq y$ 和 $\theta \in (0, 1)$ 严格成立，则称 $f$ 是严格凸函数。

#### 凹函数

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 被称为凹函数，如果 $-f$ 是凸函数。换句话说，凹函数满足：

$$f(\theta x + (1 - \theta) y) \geq \theta f(x) + (1 - \theta) f(y)$$

对于任意的 $x, y \in \text{dom}(f)$ 和 $\theta \in [0, 1]$。

#### 一阶条件

对于可微函数 $f: \mathbb{R}^n \to \mathbb{R}$（即 $f$ 在其定义域的内部可微），$f$ 是凸函数的充要条件是：其定义域 $\text{dom}(f)$ 是凸集，并且对于任意的 $x, y \in \text{dom}(f)$，有：

$$f(y) \geq f(x) + \nabla f(x)^T (y - x)$$

这个不等式的几何意义是：函数 $f$ 的图像始终位于其任意一点的切线（或切平面）上方。

#### 二阶条件

对于二阶可微函数 $f: \mathbb{R}^n \to \mathbb{R}$，$f$ 是凸函数的充要条件是：其定义域 $\text{dom}(f)$ 是凸集，并且其海森矩阵在定义域的内部处处半正定，即对于任意的 $x \in \text{int}(\text{dom}(f))$，有：

$$\nabla^2 f(x) \succeq 0$$

对于严格凸函数，海森矩阵需要是正定的，即 $\nabla^2 f(x) \succ 0$。

### 2.2.2 常见的凸函数示例

以下是一些常见的凸函数：

1. **线性函数**：$f(x) = a^T x + b$，其中 $a \in \mathbb{R}^n$，$b \in \mathbb{R}$。线性函数既是凸函数也是凹函数。

2. **二次函数**：$f(x) = (1/2) x^T P x + q^T x + r$，其中 $P \succeq 0$，$q \in \mathbb{R}^n$，$r \in \mathbb{R}$。当 $P \succ 0$ 时，二次函数是严格凸的。

3. **范数**：任意范数 $\|x\|$ 都是凸函数。

4. **最大函数**：$f(x) = \max\{x_1, x_2, \ldots, x_n\}$ 是凸函数。

5. **指数函数**：$f(x) = e^{ax}$ 对于任意的 $a \in \mathbb{R}$ 都是凸函数。

6. **负对数函数**：$f(x) = -\log x$ 在定义域 $(0, +\infty)$ 上是凸函数。

7. **负熵函数**：$f(x) = x \log x$ 在定义域 $(0, +\infty)$ 上是凸函数。

8. **矩阵迹函数**：$f(X) = \text{tr}(X)$ 是凸函数（当 $X$ 是对称矩阵时）。

9. **矩阵对数行列式**：$f(X) = -\log \det X$ 在定义域 $S_{++}^n$（所有正定矩阵的集合）上是凸函数。

### 2.2.3 保凸运算

凸函数在以下运算下保持凸性：

1. **非负加权和**：如果 $f_1, f_2, \ldots, f_k$ 都是凸函数，$w_1, w_2, \ldots, w_k \geq 0$，则 $f(x) = \sum_{i=1}^k w_i f_i(x)$ 是凸函数。

2. **复合仿射变换**：如果 $f$ 是凸函数，$g(x) = f(Ax + b)$，其中 $A \in \mathbb{R}^{m \times n}$，$b \in \mathbb{R}^m$，则 $g$ 是凸函数。

3. **逐点最大**：如果 $f_1, f_2, \ldots, f_k$ 都是凸函数，则 $f(x) = \max\{f_1(x), f_2(x), \ldots, f_k(x)\}$ 是凸函数。

4. **逐点上确界**：如果对于每个 $y \in A$，$f(x, y)$ 关于 $x$ 是凸函数，则 $g(x) = \sup_{y \in A} f(x, y)$ 是凸函数。

5. **复合函数**：设 $f: \mathbb{R}^m \to \mathbb{R}$，$g: \mathbb{R}^n \to \mathbb{R}^m$，定义复合函数 $h(x) = f(g(x))$。如果 $f$ 是凸函数且非递减，$g$ 是凸函数，则 $h$ 是凸函数；如果 $f$ 是凸函数且非递增，$g$ 是凹函数，则 $h$ 是凸函数。

### 2.2.4 拟凸函数与对数凹函数

#### 拟凸函数

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 被称为拟凸函数，如果其定义域 $\text{dom}(f)$ 是凸集，并且对于任意的 $x, y \in \text{dom}(f)$ 和任意的 $\theta \in [0, 1]$，都有：

$$f(\theta x + (1 - \theta) y) \leq \max\{f(x), f(y)\}$$

拟凸函数的水平集 $\{x \in \text{dom}(f) \mid f(x) \leq t\}$ 对于任意的 $t \in \mathbb{R}$ 都是凸集。

#### 对数凹函数

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 被称为对数凹函数，如果其定义域 $\text{dom}(f)$ 是凸集，$f(x) > 0$ 对于所有 $x \in \text{dom}(f)$，并且 $\log f(x)$ 是凹函数。

对数凹函数满足：对于任意的 $x, y \in \text{dom}(f)$ 和 $\theta \in [0, 1]$，有：

$$f(\theta x + (1 - \theta) y) \geq f(x)^\theta f(y)^{1 - \theta}$$

## 2.3 共轭函数

### 2.3.1 共轭函数的定义与性质

#### 共轭函数的定义

设 $f: \mathbb{R}^n \to \mathbb{R}$，则 $f$ 的共轭函数 $f^*: \mathbb{R}^n \to \mathbb{R}$ 定义为：

$$f^*(y) = \sup_{x \in \text{dom}(f)} (y^T x - f(x))$$

共轭函数的定义域是 $\text{dom}(f^*) = \{y \in \mathbb{R}^n \mid \sup_{x \in \text{dom}(f)} (y^T x - f(x)) < +\infty\}$。

#### 共轭函数的性质

1. **共轭函数是凸函数**：无论 $f$ 是否凸，$f^*$ 总是凸函数，因为它是一系列关于 $y$ 的线性函数的逐点上确界。

2. **Fenchel不等式**：对于任意的 $x \in \text{dom}(f)$ 和 $y \in \text{dom}(f^*)$，有：

   $$f(x) + f^*(y) \geq x^T y$$

3. **共轭的共轭**：如果 $f$ 是闭凸函数（即 $f$ 是凸函数且其 epi 图是闭集），则 $f^{**} = f$。

4. **仿射变换的共轭**：设 $g(x) = f(Ax + b)$，其中 $A \in \mathbb{R}^{m \times n}$，$b \in \mathbb{R}^m$，则 $g^*$ 为：

   $$g^*(y) = f^*(A^T y) - b^T y$$

5. **标量乘法的共轭**：设 $g(x) = \alpha f(x)$，其中 $\alpha > 0$，则 $g^*(y) = \alpha f^*(y/\alpha)$。

### 2.3.2 共轭函数的计算示例

#### 线性函数的共轭

设 $f(x) = a^T x + b$，其中 $a \in \mathbb{R}^n$，$b \in \mathbb{R}$，则：

$$f^*(y) = \sup_x (y^T x - a^T x - b) = \sup_x ((y - a)^T x - b)$$

当 $y = a$ 时，上确界为 $-b$；当 $y \neq a$ 时，上确界为 $+\infty$。因此：

$$f^*(y) = \begin{cases}
-b, & y = a \\
+\infty, & \text{otherwise}
\end{cases}$$

#### 二次函数的共轭

设 $f(x) = (1/2) x^T P x$，其中 $P \succ 0$，则：

$$f^*(y) = \sup_x (y^T x - (1/2) x^T P x)$$

对 $x$ 求导并令导数为零，得到 $y - P x = 0$，即 $x = P^{-1} y$。代入上式得：

$$f^*(y) = (1/2) y^T P^{-1} y$$

#### 负对数函数的共轭

设 $f(x) = -\log x$，定义域为 $(0, +\infty)$，则：

$$f^*(y) = \sup_{x > 0} (xy + \log x)$$

对 $x > 0$ 求导并令导数为零，得到 $y + 1/x = 0$，即 $x = -1/y$（要求 $y < 0$）。代入上式得：

$$f^*(y) = \begin{cases}
-1 - \log(-y), & y < 0 \\
+\infty, & y \geq 0
\end{cases}$$

### 2.3.3 Fenchel不等式

Fenchel不等式是共轭函数的一个重要性质，它提供了一个下界：

$$f(x) + f^*(y) \geq x^T y$$

对于任意的 $x \in \text{dom}(f)$ 和 $y \in \text{dom}(f^*)$。

当且仅当 $y = \nabla f(x)$（如果 $f$ 可微）时，等号成立。这个条件称为Fenchel对偶性条件。

### 2.3.4 小结

本章介绍了凸集与凸函数的基本概念和性质，包括：

1. **凸集**：定义、基本性质、重要示例（仿射集、多面体、半空间、球、半正定锥等）、凸集的运算以及分离定理与支撑超平面。

2. **凸函数**：定义、一阶和二阶判定条件、常见的凸函数示例、保凸运算以及拟凸函数与对数凹函数的概念。

3. **共轭函数**：定义、基本性质、计算示例以及Fenchel不等式。

这些概念和性质是凸优化的基础，它们为我们后续研究凸优化问题及其求解算法提供了重要的理论工具。在接下来的章节中，我们将学习如何将实际问题建模为凸优化问题，以及如何设计高效的算法来求解这些问题。

## 习题

### 基础题

**习题2.1**：证明集合 $C = \{x \in \mathbb{R}^n \mid \lVert x \rVert_2 \leq 1\}$ 是凸集。

**提示**：使用三角不等式。

**习题2.2**：证明函数 $f(x) = e^{ax}$ 是凸函数，其中 $a \in \mathbb{R}$。

**习题2.3**：证明两个凸集的交集仍然是凸集。

**习题2.4**：证明线性函数 $f(x) = a^T x + b$ 既是凸函数又是凹函数。

### 中等题

**习题2.5**：证明集合 $C = \{x \in \mathbb{R}^n \mid Ax = b\}$ 是凸集，其中 $A$ 是 $m \times n$ 矩阵，$b \in \mathbb{R}^m$。

**习题2.6**：证明函数 $f(x) = \lVert x \rVert_2^2$ 是凸函数。

**习题2.7**：设 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 和 $g: \mathbb{R}^n \rightarrow \mathbb{R}$ 都是凸函数，证明 $h(x) = \max\{f(x), g(x)\}$ 也是凸函数。

**习题2.8**：证明集合 $C = \{x \in \mathbb{R}^2 \mid x_1^2 + 2x_2^2 \leq 1\}$ 是凸集。

习题解答[点击这里](quiz/exercise_solutions.md)
