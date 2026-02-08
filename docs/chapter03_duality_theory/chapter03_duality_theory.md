# 第3章 凸优化的对偶理论

## 本章导读

对偶理论是凸优化的核心内容之一，它为求解复杂优化问题提供了强有力的工具。本章介绍拉格朗日对偶问题的构造方法，深入分析KKT最优性条件，并探讨对偶分解等实用算法。通过本章学习，你将理解强对偶性的深层含义，掌握利用对偶问题求解原问题的技巧，并能够将这些方法应用于机器人优化问题。

---

## 3.1 拉格朗日对偶问题

### 3.1.1 拉格朗日函数的构造

考虑标准形式的优化问题（不一定是凸的）：

$$\begin{align}
\min_{x} \quad & f_0(x) \\
\text{s.t.} \quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{align}$$

**定义 3.1（拉格朗日函数）** 上述问题的**拉格朗日函数** $L: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ 定义为：

$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

其中：
- $\lambda = (\lambda_1, \ldots, \lambda_m)^T \geq 0$ 是**不等式约束的拉格朗日乘子**（对偶变量）
- $\nu = (\nu_1, \ldots, \nu_p)^T$ 是**等式约束的拉格朗日乘子**

**直观理解**：拉格朗日函数将约束"软化"到目标函数中。当约束被违反时，拉格朗日函数会对目标函数施加惩罚（或奖励）。

**例 3.1 线性规划的拉格朗日函数**

对于 LP：

$$\begin{align}
\min_x \quad & c^T x \\
\text{s.t.} \quad & Ax = b \\
& x \succeq 0
\end{align}$$

拉格朗日函数为：

$$L(x, \lambda, \nu) = c^T x - \lambda^T x + \nu^T(Ax - b) = (c - \lambda + A^T \nu)^T x - b^T \nu$$

其中 $\lambda \succeq 0$。

### 3.1.2 拉格朗日对偶函数

**定义 3.2（对偶函数）** **拉格朗日对偶函数** $g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ 定义为拉格朗日函数关于 $x$ 的下确界：

$$g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu)$$

**对偶函数的性质**：

定理 3.1 对偶函数 $g(\lambda, \nu)$ 是凹函数（无论原问题是否是凸的）。

*证明*：$g$ 是一族关于 $(\lambda, \nu)$ 的仿射函数（对每个固定的 $x$，$L(x, \cdot, \cdot)$ 是仿射的）的下确界。仿射函数既是凸的也是凹的，凹函数的下确界仍是凹的。$\square$

定理 3.2（下界性质） 若 $\lambda \succeq 0$，则 $g(\lambda, \nu) \leq p^*$，其中 $p^*$ 是原问题的最优值。

*证明*：设 $\tilde{x}$ 是原问题的可行点，即 $f_i(\tilde{x}) \leq 0$，$h_j(\tilde{x}) = 0$。对 $\lambda \succeq 0$：

$$L(\tilde{x}, \lambda, \nu) = f_0(\tilde{x}) + \sum_{i=1}^m \lambda_i f_i(\tilde{x}) + \sum_{j=1}^p \nu_j h_j(\tilde{x}) \leq f_0(\tilde{x})$$

因此：

$$g(\lambda, \nu) = \inf_x L(x, \lambda, \nu) \leq L(\tilde{x}, \lambda, \nu) \leq f_0(\tilde{x})$$

这对所有可行点 $\tilde{x}$ 成立，故 $g(\lambda, \nu) \leq p^*$。$\square$

### 3.1.3 拉格朗日对偶问题

**定义 3.3（对偶问题）** 基于下界性质，我们希望找到最好的下界，这引出**拉格朗日对偶问题**：

$$\begin{align}
\max_{\lambda, \nu} \quad & g(\lambda, \nu) \\
\text{s.t.} \quad & \lambda \succeq 0
\end{align}$$

原问题称为** primal problem**（原问题），对偶问题称为**dual problem**（对偶问题）。

**对偶问题的性质**：

- 对偶问题总是凸优化问题（最大化凹函数等价于最小化凸函数）
- 即使原问题非凸，对偶问题也是凸的
- 对偶问题提供了原问题最优值的下界

**例 3.2 线性规划的对偶**

对于标准形式 LP：

$$\begin{align}
\min_x \quad & c^T x \\
\text{s.t.} \quad & Ax = b \\
& x \succeq 0
\end{align}$$

对偶函数为：

$$g(\lambda, \nu) = \inf_x [(c - \lambda + A^T \nu)^T x - b^T \nu] = \begin{cases} -b^T \nu & \text{if } c - \lambda + A^T \nu = 0 \\ -\infty & \text{otherwise} \end{cases}$$

因此对偶问题为：

$$\begin{align}
\max_{\lambda, \nu} \quad & -b^T \nu \\
\text{s.t.} \quad & c - \lambda + A^T \nu = 0 \\
& \lambda \succeq 0
\end{align}$$

等价于：

$$\begin{align}
\max_{\nu} \quad & b^T \nu \\
\text{s.t.} \quad & A^T \nu \preceq c
\end{align}$$

**例 3.3 最小二乘问题的对偶**

考虑带约束的最小二乘：

$$\begin{align}
\min_x \quad & \frac{1}{2}\|Ax - b\|_2^2 \\
\text{s.t.} \quad & x \succeq 0
\end{align}$$

拉格朗日函数：

$$L(x, \lambda) = \frac{1}{2}\|Ax - b\|_2^2 - \lambda^T x$$

对偶函数：

$$g(\lambda) = \inf_x L(x, \lambda)$$

令 $\nabla_x L = A^T(Ax - b) - \lambda = 0$，得 $x = (A^T A)^{-1}(A^T b + \lambda)$（假设 $A^T A$ 可逆）。代入得：

$$g(\lambda) = -\frac{1}{2}\lambda^T (A^T A)^{-1} \lambda - b^T A(A^T A)^{-1} \lambda + \text{const}$$

对偶问题是关于 $\lambda$ 的二次规划。

### 3.1.4 弱对偶性与强对偶性

**定义 3.4（对偶间隙）** 设 $p^*$ 是原问题的最优值，$d^*$ 是对偶问题的最优值。**对偶间隙**定义为 $p^* - d^* \geq 0$。

**弱对偶性**

定理 3.3（弱对偶性） 对于任何优化问题（不一定是凸的），总有 $d^* \leq p^*$，即对偶间隙非负。

*证明*：由对偶函数的下界性质，对所有可行 $(\lambda, \nu)$ 有 $g(\lambda, \nu) \leq p^*$。因此 $d^* = \max g(\lambda, \nu) \leq p^*$。$\square$

弱对偶性即使对于非凸问题也成立，这使得对偶方法在非凸优化中也有应用价值。

**强对偶性**

定义 3.5（强对偶性） 若 $d^* = p^*$，则称**强对偶性**成立。

强对偶性不总是成立，但对于凸优化问题，在 mild 条件下成立。

**Slater 条件**

定理 3.4（Slater 条件） 对于凸优化问题，若存在严格可行点 $x$ 满足：

$$f_i(x) < 0, \quad i = 1, \ldots, m, \quad Ax = b$$

则强对偶性成立。（对于仿射不等式约束，只需 $f_i(x) \leq 0$）

Slater 条件是一种**约束品性**（constraint qualification），保证了强对偶性。

**几何解释**

考虑只有一个不等式约束的问题：

$$\min_x f_0(x) \quad \text{s.t.} \quad f_1(x) \leq 0$$

定义集合：

$$\mathcal{G} = \{(f_1(x), f_0(x)) \mid x \in D\}$$

原问题最优值 $p^* = \inf\{t \mid (u, t) \in \mathcal{G}, u \leq 0\}$。

对偶函数 $g(\lambda) = \inf\{\lambda u + t \mid (u, t) \in \mathcal{G}\}$。

对偶问题是在寻找支撑超平面，使得 $p^* - d^*$ 最小化。当 $\mathcal{G}$ 是凸集时，强对偶性通常成立。

---

## 3.2 KKT条件

### 3.2.1 KKT条件的推导

**Karush-Kuhn-Tucker (KKT) 条件**是最优性条件的重要组成部分，它结合了一阶最优性条件、原始可行性、对偶可行性和互补松弛性。

**KKT条件的内容**

对于优化问题（假设可微）：

$$\begin{align}
\min_x \quad & f_0(x) \\
\text{s.t.} \quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{align}$$

**KKT条件**包括：

1. **原始可行性**（Primal feasibility）：
   $$f_i(x^*) \leq 0, \quad i = 1, \ldots, m$$
   $$h_j(x^*) = 0, \quad j = 1, \ldots, p$$

2. **对偶可行性**（Dual feasibility）：
   $$\lambda^* \succeq 0$$

3. **互补松弛性**（Complementary slackness）：
   $$\lambda_i^* f_i(x^*) = 0, \quad i = 1, \ldots, m$$

4. **平稳性**（Stationarity）：
   $$\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$

### 3.2.2 KKT条件的必要性

定理 3.5（KKT条件的必要性） 对于可微的凸优化问题，若强对偶性成立，则 $x^*$ 和 $(\lambda^*, \nu^*)$ 分别是原问题和对偶问题的最优解当且仅当它们满足 KKT 条件。

*证明*：

（必要性）设 $x^*$ 和 $(\lambda^*, \nu^*)$ 是最优解且强对偶性成立，即 $f_0(x^*) = g(\lambda^*, \nu^*)$。

由对偶函数定义：

$$g(\lambda^*, \nu^*) = \inf_x L(x, \lambda^*, \nu^*) \leq L(x^*, \lambda^*, \nu^*) \leq f_0(x^*)$$

等式成立意味着：
- $x^*$ 最小化 $L(x, \lambda^*, \nu^*)$，故平稳性条件成立
- $L(x^*, \lambda^*, \nu^*) = f_0(x^*)$，即 $\sum \lambda_i^* f_i(x^*) = 0$，由 $\lambda^* \succeq 0$ 和 $f_i(x^*) \leq 0$ 得互补松弛性

（充分性）设 KKT 条件成立。由平稳性，$x^*$ 最小化 $L(x, \lambda^*, \nu^*)$，故：

$$g(\lambda^*, \nu^*) = L(x^*, \lambda^*, \nu^*) = f_0(x^*) + \sum \lambda_i^* f_i(x^*) = f_0(x^*)$$

由互补松弛性。因此 $d^* \geq g(\lambda^*, \nu^*) = f_0(x^*) \geq p^*$，结合弱对偶性 $d^* \leq p^*$，得强对偶性成立且 $x^*$ 和 $(\lambda^*, \nu^*)$ 是最优解。$\square$

### 3.2.3 KKT条件的充分性

对于凸优化问题，KKT 条件也是充分的：

定理 3.6（KKT条件的充分性） 对于可微的凸优化问题，若 $(x^*, \lambda^*, \nu^*)$ 满足 KKT 条件，则 $x^*$ 是原问题的最优解，$(\lambda^*, \nu^*)$ 是对偶问题的最优解，且强对偶性成立。

*证明*：由凸性，平稳性条件意味着 $x^*$ 是拉格朗日函数的最小值点。其余同上。$\square$

### 3.2.4 互补松弛性的几何意义

互补松弛性条件 $\lambda_i^* f_i(x^*) = 0$ 意味着：
- 若 $\lambda_i^* > 0$，则 $f_i(x^*) = 0$（约束是紧的，active）
- 若 $f_i(x^*) < 0$，则 $\lambda_i^* = 0$（约束是松的，inactive）

几何解释：拉格朗日乘子可以看作约束的"价格"或"影子价格"。若约束不紧（有松弛），则其价格为零；若约束是紧的，则其价格可能非零。

### 3.2.5 KKT条件在机器人优化中的应用

**例 3.4 机器人接触力优化**

考虑多接触机器人的力分配问题：

$$\begin{align}
\min_{f} \quad & \|f\|_2^2 \\
\text{s.t.} \quad & Gf = w \\
& f \in \mathcal{F}
\end{align}$$

其中 $f$ 是接触力，$G$ 是抓取矩阵，$w$ 是期望 wrench，$\mathcal{F}$ 是摩擦锥。

KKT 条件可以用来分析最优力的结构，以及设计高效的求解算法。

---

## 3.3 对偶问题的求解方法

### 3.3.1 对偶上升法

**基本思想**：交替更新原变量和对偶变量。

算法 3.1（对偶上升法）

重复直到收敛：
1. $x^{k+1} = \arg\min_x L(x, \lambda^k, \nu^k)$
2. $\lambda_i^{k+1} = \lambda_i^k + \alpha_k f_i(x^{k+1})$，$i = 1, \ldots, m$
3. $\nu_j^{k+1} = \nu_j^k + \alpha_k h_j(x^{k+1})$，$j = 1, \ldots, p$

其中 $\alpha_k > 0$ 是步长。

**收敛性**：在 mild 条件下，对偶上升法收敛到对偶最优解。

**缺点**：
- 需要精确求解关于 $x$ 的最小化问题，计算代价高
- 收敛速度可能较慢

### 3.3.2 增广拉格朗日方法

为了克服对偶上升法的缺点，引入增广拉格朗日函数：

**定义 3.6（增广拉格朗日函数）** 对于等式约束问题，增广拉格朗日函数为：

$$L_\rho(x, \nu) = f_0(x) + \nu^T h(x) + \frac{\rho}{2}\|h(x)\|_2^2$$

其中 $\rho > 0$ 是惩罚参数。

算法 3.2（增广拉格朗日法/乘子法）

重复直到收敛：
1. $x^{k+1} = \arg\min_x L_\rho(x, \nu^k)$
2. $\nu^{k+1} = \nu^k + \rho h(x^{k+1})$

**优点**：
- 增广项 $\frac{\rho}{2}\|h(x)\|_2^2$ 使得子问题更容易求解（更好的条件数）
- 对偶更新使用固定步长 $\rho$
- 收敛速度更快

**扩展到不等式约束**：

对于不等式约束，可以引入松弛变量转化为等式约束：

$$f_i(x) + s_i = 0, \quad s_i \geq 0$$

### 3.3.3 交替方向乘子法（ADMM）

ADMM 是增广拉格朗日方法的扩展，适用于可分离的问题：

$$\begin{align}
\min_{x, z} \quad & f(x) + g(z) \\
\text{s.t.} \quad & Ax + Bz = c
\end{align}$$

算法 3.3（ADMM）

重复直到收敛：
1. $x^{k+1} = \arg\min_x L_\rho(x, z^k, \nu^k)$
2. $z^{k+1} = \arg\min_z L_\rho(x^{k+1}, z, \nu^k)$
3. $\nu^{k+1} = \nu^k + \rho(Ax^{k+1} + Bz^{k+1} - c)$

**ADMM 的优势**：
- 将原问题分解为两个（通常更简单的）子问题
- 适用于分布式优化
- 在大规模机器学习和统计学习中有广泛应用

**例 3.5 LASSO 问题的 ADMM 求解**

LASSO 问题：

$$\min_x \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1$$

引入辅助变量 $z = x$：

$$\begin{align}
\min_{x, z} \quad & \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|z\|_1 \\
\text{s.t.} \quad & x - z = 0
\end{align}$$

ADMM 迭代：
1. $x^{k+1} = (A^T A + \rho I)^{-1}(A^T b + \rho(z^k - \nu^k))$
2. $z^{k+1} = S_{\lambda/\rho}(x^{k+1} + \nu^k)$（软阈值算子）
3. $\nu^{k+1} = \nu^k + x^{k+1} - z^{k+1}$

### 3.3.4 对偶分解

对偶分解适用于可分离的问题：

$$\begin{align}
\min_x \quad & \sum_{i=1}^N f_i(x_i) \\
\text{s.t.} \quad & \sum_{i=1}^N A_i x_i = b
\end{align}$$

拉格朗日函数：

$$L(x, \nu) = \sum_{i=1}^N f_i(x_i) + \nu^T(\sum_{i=1}^N A_i x_i - b) = \sum_{i=1}^N [f_i(x_i) + \nu^T A_i x_i] - \nu^T b$$

对偶函数分解为 $N$ 个独立的子问题：

$$g(\nu) = \sum_{i=1}^N \inf_{x_i} [f_i(x_i) + \nu^T A_i x_i] - \nu^T b$$

算法 3.4（对偶分解）

重复直到收敛：
1. 对每个 $i$，并行求解：$x_i^{k+1} = \arg\min_{x_i} [f_i(x_i) + (\nu^k)^T A_i x_i]$
2. 更新对偶变量：$\nu^{k+1} = \nu^k + \alpha_k (\sum_i A_i x_i^{k+1} - b)$

**应用**：分布式优化、多智能体系统、资源分配等。

---

## 3.4 本章小结

本章深入探讨了凸优化的对偶理论：

1. **拉格朗日对偶**：通过拉格朗日函数构造对偶问题，对偶函数总是凹的，提供原问题最优值的下界。

2. **弱对偶与强对偶**：
   - 弱对偶性：$d^* \leq p^*$ 总是成立
   - 强对偶性：$d^* = p^*$ 在凸问题和 Slater 条件下成立

3. **KKT条件**：最优解的必要（对于凸问题也是充分）条件，包括原始可行性、对偶可行性、互补松弛性和平稳性。

4. **对偶求解方法**：
   - 对偶上升法：交替更新原变量和对偶变量
   - 增广拉格朗日法：通过增广项改善收敛性
   - ADMM：适用于可分离问题，便于分布式计算
   - 对偶分解：将大规模问题分解为子问题

对偶理论不仅提供了求解优化问题的有效工具，还揭示了原问题和对偶问题之间的深刻联系，在灵敏度分析、算法设计和分布式优化中有重要应用。

---

## 习题

### 基础题

3.1 推导以下问题的对偶问题：
   (a) 二次规划：$\min_x \frac{1}{2}x^T P x + q^T x$，s.t. $Ax \leq b$
   (b) $l_1$ 正则化最小二乘：$\min_x \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1$

3.2 验证以下问题满足 Slater 条件：
   $$\min_x x^2 \quad \text{s.t.} \quad x \geq 1$$

3.3 写出以下问题的 KKT 条件：
   $$\min_x \frac{1}{2}\|x\|_2^2 \quad \text{s.t.} \quad Ax = b, \quad x \succeq 0$$

3.4 证明：对于 LP，原问题和对偶问题的最优值相等（强对偶性）。

### 提高题

3.5 考虑支持向量机（SVM）优化问题：
   $$\min_{w, b} \frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^n \max(0, 1 - y_i(w^T x_i + b))$$
   推导其对偶问题，并解释对偶变量的几何意义。

3.6 证明：对于凸二次规划，若 $P \succ 0$ 且可行集非空，则强对偶性成立。

3.7 设计 ADMM 算法求解以下问题：
   $$\min_x \|Ax - b\|_1 + \lambda\|x\|_2^2$$

3.8 考虑分布式优化问题：
   $$\min_x \sum_{i=1}^N f_i(x)$$
   其中每个 $f_i$ 只有第 $i$ 个节点知道。设计基于对偶分解的分布式算法。

### 编程题

3.9 实现增广拉格朗日法求解等式约束 QP，并与直接求解 KKT 系统比较计算效率。

3.10 实现 ADMM 求解 LASSO 问题，并在真实数据集上测试。

---

## 参考文献

1. Boyd S, Vandenberghe L. Convex Optimization[M]. Cambridge University Press, 2004.
2. Bertsekas D P. Nonlinear Programming[M]. Athena Scientific, 1999.
3. Nocedal J, Wright S. Numerical Optimization[M]. Springer, 2006.
4. Boyd S, Parikh N, Chu E, et al. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers[J]. Foundations and Trends in Machine Learning, 2011.
