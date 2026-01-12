# 第1章 介绍与预备知识

## 1.1 课程介绍

### 1.1.1 什么是最优化？

最优化是数学的一个重要分支，它研究如何在给定的约束条件下，找到使目标函数达到最大值或最小值的变量取值。简单来说，最优化就是"寻找最好的选择"的数学方法。

在数学上，一个典型的优化问题可以表示为：

$$\min_{x} f(x)$$

$$\text{s.t. } g_i(x) \leq 0, \quad i=1,2,\ldots,m$$

$$h_j(x) = 0, \quad j=1,2,\ldots,p$$

其中：
- $f(x)$ 是目标函数（我们要最小化的函数）
- $g_i(x) \leq 0$ 是不等式约束
- $h_j(x) = 0$ 是等式约束
- $x$ 是优化变量

### 1.1.2 凸优化的重要性与应用领域

凸优化是最优化的一个重要子类，它研究的是目标函数为凸函数、约束集合为凸集的优化问题。凸优化之所以重要，主要有以下几个原因：

1. **理论上的优良性质**：凸优化问题的局部最优解就是全局最优解，这使得求解变得相对容易。
2. **算法的有效性**：存在许多高效的算法可以求解大规模的凸优化问题。
3. **广泛的应用**：凸优化在许多领域都有重要应用，包括：
   - 机器学习与人工智能
   - 信号处理与通信
   - 控制系统
   - 金融工程
   - 运筹学
   - 网络优化
   - 科学计算

### 1.1.3 课程内容与学习目标

本课程旨在为具备基础数学知识的学生提供凸优化与最优化的入门介绍。通过本课程的学习，你将：

1. **掌握基本概念**：理解凸集、凸函数、凸优化问题等基本概念。
2. **学习优化算法**：掌握无约束和约束优化的主要算法。
3. **理解对偶理论**：掌握拉格朗日对偶理论及其应用。
4. **解决实际问题**：能够将实际问题建模为凸优化问题并求解。
5. **应用软件工具**：学会使用常见的优化软件工具。

## 1.2 数学预备知识

本节将回顾本课程所需的数学基础知识，包括向量与矩阵运算、多元函数微积分、线性代数基础和概率论基础。

### 1.2.1 向量与矩阵运算复习

#### 向量运算

- **向量加法**：对于向量 $x, y \in \mathbb{R}^n$，有 $x + y = (x_1 + y_1, x_2 + y_2, \ldots, x_n + y_n)^T$。

- **标量乘法**：对于向量 $x \in \mathbb{R}^n$ 和标量 $\alpha \in \mathbb{R}$，有 $\alpha x = (\alpha x_1, \alpha x_2, \ldots, \alpha x_n)^T$。

- **点积**：对于向量 $x, y \in \mathbb{R}^n$，点积定义为 $x^T y = \sum_{i=1}^n x_i y_i$。

- **范数**：
  - 欧几里得范数（2-范数）：$\lVert x \rVert_2 = \sqrt{x^T x} = \sqrt{\sum_{i=1}^n x_i^2}$
  - 1-范数：$\lVert x \rVert_1 = \sum_{i=1}^n |x_i|$
  - 无穷范数：$\lVert x \rVert_\infty = \max_{1 \leq i \leq n} |x_i|$

#### 矩阵运算

- **矩阵加法**：对于矩阵 $A, B \in \mathbb{R}^{m \times n}$，设 $a_{ij}$ 和 $b_{ij}$ 分别是 $A$ 和 $B$ 的第 $i$ 行第 $j$ 列元素，则矩阵和 $C = A + B$ 的元素为 $c_{ij} = a_{ij} + b_{ij}$。

- **标量乘法**：对于矩阵 $A \in \mathbb{R}^{m \times n}$ 和标量 $\alpha \in \mathbb{R}$，设 $a_{ij}$ 是 $A$ 的第 $i$ 行第 $j$ 列元素，则标量乘法 $B = \alpha A$ 的元素为 $b_{ij} = \alpha a_{ij}$。

- **矩阵乘法**：对于矩阵 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，设 $a_{ij}$ 和 $b_{jk}$ 分别是 $A$ 和 $B$ 的元素，则矩阵乘积 $C = AB$ 的元素为 $c_{ik} = \sum_{j=1}^n a_{ij} b_{jk}$。

- **转置**：对于矩阵 $A \in \mathbb{R}^{m \times n}$，设 $a_{ji}$ 是 $A$ 的第 $j$ 行第 $i$ 列元素，则其转置 $A^T \in \mathbb{R}^{n \times m}$ 的第 $i$ 行第 $j$ 列元素为 $a_{ji}$。

- **迹**：对于方阵 $A \in \mathbb{R}^{n \times n}$，其迹定义为 $\text{tr}(A) = \sum_{i=1}^n A_{ii}$。

- **行列式**：对于方阵 $A \in \mathbb{R}^{n \times n}$，其行列式记为 $\det(A)$，是一个标量值。

### 1.2.2 多元函数微积分

#### 梯度

对于函数 $f: \mathbb{R}^n \to \mathbb{R}$，其梯度是一个 $n$ 维向量，定义为：

$$\nabla f(x) = \left( \frac{\partial f(x)}{\partial x_1}, \frac{\partial f(x)}{\partial x_2}, \ldots, \frac{\partial f(x)}{\partial x_n} \right)^T$$

梯度的重要性质：梯度指向函数值增长最快的方向，负梯度指向函数值下降最快的方向。

#### 海森矩阵

对于二阶可导的函数 $f: \mathbb{R}^n \to \mathbb{R}$，其海森矩阵是一个 $n \times n$ 的对称矩阵，定义为：

$$
\nabla^2 f(x) = 
\left[
\begin{array}{cccc}
\frac{\partial^2 f(x)}{\partial x_1^2} & \frac{\partial^2 f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_n} \\\\
\frac{\partial^2 f(x)}{\partial x_2 \partial x_1} & \frac{\partial^2 f(x)}{\partial x_2^2} & \cdots & \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\frac{\partial^2 f(x)}{\partial x_n \partial x_1} & \frac{\partial^2 f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_n^2}
\end{array}
\right]
$$

#### 泰勒展开

多元函数的一阶泰勒展开：

$$f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x$$

多元函数的二阶泰勒展开：

$$f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T \nabla^2 f(x) \Delta x$$

#### 链式法则

对于复合函数 $f(g(x))$，其梯度为：

$$\nabla f(g(x)) = \nabla g(x)^T \nabla f(g(x))$$

### 1.2.3 线性代数基础

#### 特征值与特征向量

对于方阵 $A \in \mathbb{R}^{n \times n}$，如果存在标量 $\lambda$ 和非零向量 $v$ 满足：

$$Av = \lambda v$$

则称 $\lambda$ 是 $A$ 的一个特征值，$v$ 是对应的特征向量。

#### 矩阵的正定性

对于对称方阵 $A \in \mathbb{R}^{n \times n}$：
- 如果对于所有非零向量 $x \in \mathbb{R}^n$，都有 $x^T A x > 0$，则称 $A$ 是正定的，记为 $A \succ 0$。
- 如果对于所有非零向量 $x \in \mathbb{R}^n$，都有 $x^T A x \geq 0$，则称 $A$ 是半正定的，记为 $A \succeq 0$。
- 类似地，可以定义负定和半负定矩阵。

#### 奇异值分解（SVD）

对于任意矩阵 $A \in \mathbb{R}^{m \times n}$，其奇异值分解为：

$$A = U \Sigma V^T$$

其中：
- $U \in \mathbb{R}^{m \times m}$ 是正交矩阵
- $V \in \mathbb{R}^{n \times n}$ 是正交矩阵
- $\Sigma \in \mathbb{R}^{m \times n}$ 是对角矩阵，对角线上的元素称为奇异值，非负且按降序排列

#### 矩阵的秩

矩阵的秩是矩阵中线性无关的行（或列）的最大数量，记为 $\text{rank}(A)$。对于奇异值分解 $A = U \Sigma V^T$，$\text{rank}(A)$ 等于非零奇异值的个数。

### 1.2.4 概率论基础

#### 概率分布

- **期望**：对于随机变量 $X$，其期望（均值）定义为 $\mathbb{E}[X] = \int x p(x) dx$（连续型）或 $\sum x p(x)$（离散型）。

- **方差**：对于随机变量 $X$，其方差定义为 $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$。

- **协方差**：对于随机变量 $X$ 和 $Y$，其协方差定义为 $\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$。

#### 多元正态分布

多元正态分布（也称为高斯分布）是最常用的多元概率分布之一，其概率密度函数为：

$$p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

其中：
- $\mu \in \mathbb{R}^n$ 是均值向量
- $\Sigma \in \mathbb{R}^{n \times n}$ 是协方差矩阵，正定

#### 大数定律与中心极限定理

- **大数定律**：随着样本容量的增加，样本均值收敛到总体均值。
- **中心极限定理**：随着样本容量的增加，样本均值的分布近似于正态分布。

### 1.2.5 小结

本章介绍了最优化的基本概念、凸优化的重要性与应用领域，以及本课程的学习目标。同时，我们复习了课程所需的数学基础知识，包括向量与矩阵运算、多元函数微积分、线性代数基础和概率论基础。这些知识将为我们后续学习凸集、凸函数、凸优化问题及求解算法奠定基础。

在接下来的章节中，我们将开始系统学习凸优化的核心概念和方法，从凸集与凸函数的基础理论开始，逐步深入到优化算法和应用案例。

## 习题

### 基础题

**习题1.1**：证明向量的2-范数满足三角不等式，即 $\lVert x + y \rVert_2 \leq \lVert x \rVert_2 + \lVert y \rVert_2$。

**提示**：使用柯西-施瓦茨不等式 $(x^T y)^2 \leq (x^T x)(y^T y)$。

**习题1.2**：计算函数 $f(x) = x^3 - 3x$ 的梯度和海森矩阵。

**习题1.3**：证明向量的1-范数和∞-范数都满足三角不等式。

**习题1.4**：计算函数 $f(x) = \frac{1}{2}x^T Ax + b^T x + c$ 的梯度和海森矩阵，其中 $A$ 是对称矩阵。

### 中等题

**习题1.5**：设 $x, y \in \mathbb{R}^n$，证明 $\lVert x + y \rVert_2^2 + \lVert x - y \rVert_2^2 = 2(\lVert x \rVert_2^2 + \lVert y \rVert_2^2)$。

**习题1.6**：设 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 是二次可微函数，证明 $f$ 是凸函数当且仅当它的海森矩阵在定义域上半正定。

**习题1.7**：计算函数 $f(x) = \ln(1 + e^{x^T x})$ 的梯度。

习题解答[点击这里](quiz/exercise_solutions.md)
