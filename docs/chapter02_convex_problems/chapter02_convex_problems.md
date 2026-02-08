# 第2章 凸优化问题

## 本章导读

凸优化问题是凸分析理论的核心应用。本章介绍凸优化问题的标准形式、分类方法以及转化技巧，并重点讨论如何将实际问题建模为凸优化问题。通过本章学习，你将掌握线性规划、二次规划、半定规划等标准问题的形式，理解凸松弛技术，并学会将机器人路径规划、控制等问题转化为凸优化问题求解。

---

## 2.1 凸优化问题的定义和标准形式

### 2.1.1 凸优化问题的一般形式

**定义 2.1（凸优化问题）** 一个优化问题称为**凸优化问题**，如果它具有以下形式：

$$\begin{align}
\min_{x} \quad & f_0(x) \\
\text{s.t.} \quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{align}$$

其中：
- $f_0: \mathbb{R}^n \to \mathbb{R}$ 是**凸目标函数**
- $f_i: \mathbb{R}^n \to \mathbb{R}$（$i = 1, \ldots, m$）是**凸不等式约束函数**
- $h_j: \mathbb{R}^n \to \mathbb{R}$（$j = 1, \ldots, p$）是**仿射等式约束函数**：$h_j(x) = a_j^T x - b_j$

**重要说明**：凸优化问题要求等式约束必须是仿射的。如果等式约束是非线性的（即使是凸函数），问题可能不是凸优化问题。例如，$x^2 + y^2 = 1$ 定义了单位圆，虽然 $x^2 + y^2$ 是凸函数，但等式约束使得可行集非凸。

**定义域**：优化问题的定义域为：

$$D = \bigcap_{i=0}^m \text{dom } f_i \cap \bigcap_{j=1}^p \text{dom } h_j$$

**可行集**：满足所有约束的点的集合称为**可行集**（feasible set）：

$$\mathcal{X} = \{x \in D \mid f_i(x) \leq 0, i=1,\ldots,m; \quad h_j(x) = 0, j=1,\ldots,p\}$$

定理 2.1 凸优化问题的可行集是凸集。

*证明*：
- 每个不等式约束 $f_i(x) \leq 0$ 定义了凸函数 $f_i$ 的水平集，因此是凸集
- 每个等式约束 $h_j(x) = 0$（仿射）定义了仿射子空间，是凸集
- 可行集是这些凸集的交集，因此是凸集 $\square$

### 2.1.2 局部最优与全局最优

**定义 2.2（最优解）** 点 $x^*$ 称为凸优化问题的**最优解**（或全局最优解），如果 $x^* \in \mathcal{X}$ 且对所有 $x \in \mathcal{X}$ 有 $f_0(x^*) \leq f_0(x)$。

**定义 2.3（局部最优解）** 点 $x^*$ 称为**局部最优解**，如果 $x^* \in \mathcal{X}$ 且存在 $R > 0$ 使得对所有满足 $\|x - x^*\|_2 \leq R$ 的 $x \in \mathcal{X}$ 有 $f_0(x^*) \leq f_0(x)$。

**定理 2.2（凸优化的重要性质）** 对于凸优化问题，任何局部最优解都是全局最优解。

*证明*：设 $x^*$ 是局部最优解，假设存在 $y \in \mathcal{X}$ 使得 $f_0(y) < f_0(x^*)$。考虑凸组合 $z = \theta x^* + (1-\theta)y$，其中 $\theta \in (0, 1]$ 足够接近 1 使得 $\|z - x^*\|_2 \leq R$。

由于可行集是凸的，$z \in \mathcal{X}$。由于 $f_0$ 是凸函数：

$$f_0(z) \leq \theta f_0(x^*) + (1-\theta)f_0(y) < \theta f_0(x^*) + (1-\theta)f_0(x^*) = f_0(x^*)$$

这与 $x^*$ 的局部最优性矛盾。$\square$

**推论**：对于凸优化问题，如果目标函数是严格凸的，则最优解（如果存在）是唯一的。

### 2.1.3 标准形式的等价表示

凸优化问题有多种等价表示形式：

**1. 紧凑矩阵形式**

$$\begin{align}
\min_{x} \quad & f_0(x) \\
\text{s.t.} \quad & f(x) \preceq 0 \\
& Ax = b
\end{align}$$

其中 $f(x) = (f_1(x), \ldots, f_m(x))^T$，$\preceq$ 表示分量-wise 不等式。

**2. 集合约束形式**

$$\min_{x \in \mathcal{X}} f_0(x)$$

其中 $\mathcal{X}$ 是凸集。

**3. 无约束形式**

通过指示函数，可以将约束问题转化为无约束问题：

$$\min_x f_0(x) + \sum_{i=1}^m I_{\{f_i \leq 0\}}(x) + \sum_{j=1}^p I_{\{h_j = 0\}}(x)$$

其中 $I_C(x)$ 是集合 $C$ 的指示函数。

---

## 2.2 标准凸优化问题

### 2.2.1 线性规划（Linear Programming, LP）

**定义 2.4（线性规划）** 目标函数和约束函数都是仿射的凸优化问题称为**线性规划**：

$$\begin{align}
\min_{x} \quad & c^T x + d \\
\text{s.t.} \quad & Gx \preceq h \\
& Ax = b
\end{align}$$

其中 $c \in \mathbb{R}^n$，$d \in \mathbb{R}$，$G \in \mathbb{R}^{m \times n}$，$h \in \mathbb{R}^m$，$A \in \mathbb{R}^{p \times n}$，$b \in \mathbb{R}^p$。

**标准形式**：

$$\begin{align}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & Ax = b \\
& x \succeq 0
\end{align}$$

任何线性规划都可以转化为标准形式。

**不等式形式**：

$$\begin{align}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & Ax \preceq b
\end{align}$$

**几何解释**：线性规划的可行集是多面体，最优解位于多面体的顶点（如果存在）。

**求解方法**：
- 单纯形法（Simplex method）
- 内点法（Interior point method）
- 椭球法（Ellipsoid method）

### 2.2.2 二次规划（Quadratic Programming, QP）

**定义 2.5（二次规划）** 目标函数是凸二次函数、约束是仿射的凸优化问题称为**二次规划**：

$$\begin{align}
\min_{x} \quad & \frac{1}{2}x^T P x + q^T x + r \\
\text{s.t.} \quad & Gx \preceq h \\
& Ax = b
\end{align}$$

其中 $P \in \mathbb{S}_+^n$（对称半正定矩阵），$q \in \mathbb{R}^n$，$r \in \mathbb{R}$。

**特殊情形**：
- 当 $P = 0$ 时，QP 退化为 LP
- 当没有约束时，称为**无约束二次规划**

**求解方法**：
- 积极集法（Active set method）
- 内点法
- 梯度投影法

**例 2.1 最小二乘问题**

最小二乘问题是最简单的二次规划：

$$\min_x \|Ax - b\|_2^2 = (Ax - b)^T(Ax - b) = x^T A^T A x - 2b^T A x + b^T b$$

其中 $P = 2A^T A$，$q = -2A^T b$。如果 $A$ 列满秩，则 $P \succ 0$，问题有唯一解 $x^* = (A^T A)^{-1} A^T b$。

**带约束的最小二乘**：

$$\begin{align}
\min_{x} \quad & \|Ax - b\|_2^2 \\
\text{s.t.} \quad & Cx = d
\end{align}$$

可以用拉格朗日乘子法求解。

### 2.2.3 二次约束二次规划（QCQP）

**定义 2.6（QCQP）** 目标函数和不等式约束都是凸二次函数的优化问题：

$$\begin{align}
\min_{x} \quad & \frac{1}{2}x^T P_0 x + q_0^T x + r_0 \\
\text{s.t.} \quad & \frac{1}{2}x^T P_i x + q_i^T x + r_i \leq 0, \quad i = 1, \ldots, m \\
& Ax = b
\end{align}$$

其中 $P_i \in \mathbb{S}_+^n$（$i = 0, 1, \ldots, m$）。

**几何解释**：QCQP 的可行集是椭球的交集（与仿射子空间的交集）。

### 2.2.4 半定规划（Semidefinite Programming, SDP）

**定义 2.7（半定规划）** 涉及半正定矩阵锥的凸优化问题称为**半定规划**：

$$\begin{align}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & x_1 F_1 + x_2 F_2 + \cdots + x_n F_n + G \preceq 0 \\
& Ax = b
\end{align}$$

其中 $F_i, G \in \mathbb{S}^k$（对称矩阵），$\preceq$ 表示矩阵的半定序（即 $X \preceq Y$ 当且仅当 $Y - X$ 半正定）。

**矩阵形式**：

$$\begin{align}
\min_{X} \quad & \text{tr}(CX) \\
\text{s.t.} \quad & \text{tr}(A_i X) = b_i, \quad i = 1, \ldots, p \\
& X \succeq 0
\end{align}$$

其中 $C, A_i, X \in \mathbb{S}^n$。

**SDP 的表达能力**：

定理 2.3 LP 和 QP 都可以表示为 SDP 的特例。

*证明思路*：
- 对于 LP，对角矩阵 $X = \text{diag}(x)$，约束 $x \succeq 0$ 等价于 $X \succeq 0$
- 对于 QP，可以通过引入辅助变量和 Schur 补转化为 SDP $\square$

**例 2.2 特征值优化**

最小化对称矩阵的最大特征值：

$$\min_x \lambda_{\max}(A(x))$$

其中 $A(x) = A_0 + x_1 A_1 + \cdots + x_n A_n$，$A_i \in \mathbb{S}^m$。

这可以转化为 SDP：

$$\begin{align}
\min_{x, t} \quad & t \\
\text{s.t.} \quad & A(x) \preceq tI
\end{align}$$

### 2.2.5 锥规划（Conic Programming）

**定义 2.8（锥规划）** 一般的锥规划形式为：

$$\begin{align}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & Ax = b \\
& x \in K
\end{align}$$

其中 $K$ 是闭凸锥。

**特殊情形**：
- $K = \mathbb{R}_+^n$：线性规划
- $K = \mathcal{Q}^n$（二阶锥）：二阶锥规划（SOCP）
- $K = \mathbb{S}_+^n$：半定规划

**二阶锥规划（SOCP）**：

$$\begin{align}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & \|A_i x + b_i\|_2 \leq c_i^T x + d_i, \quad i = 1, \ldots, m \\
& Fx = g
\end{align}$$

SOCP 可以处理鲁棒优化、投资组合优化等问题。

---

## 2.3 凸优化问题的转化

### 2.3.1 等价变换

**定义 2.9（问题等价）** 两个优化问题称为**等价**，如果从一个问题的解可以容易地得到另一个问题的解，反之亦然。

**常见等价变换**：

**1. 变量替换**

设 $\phi: \mathbb{R}^n \to \mathbb{R}^n$ 是一一映射，则问题：

$$\min_x f(x) \quad \text{s.t.} \quad x \in \mathcal{X}$$

等价于：

$$\min_y f(\phi(y)) \quad \text{s.t.} \quad \phi(y) \in \mathcal{X}$$

**例 2.3 几何规划**

几何规划通过变量替换 $y_i = \log x_i$ 可以转化为凸优化问题。

**2. 目标函数变换**

- 若 $\phi: \mathbb{R} \to \mathbb{R}$ 是单调递增函数，则 $\min f_0(x)$ 与 $\min \phi(f_0(x))$ 有相同的最优解集
- 例如，$\min \|Ax - b\|_2$ 与 $\min \|Ax - b\|_2^2$ 等价

**3. 约束变换**

- 等式约束 $h(x) = 0$ 可以表示为两个不等式约束：$h(x) \leq 0$ 和 $-h(x) \leq 0$
- 但只有当 $h$ 是仿射时，这种表示才保持凸性

### 2.3.2 松弛与限制

**松弛（Relaxation）**

定义 2.10（松弛问题） 问题 $p^*_{\text{rel}}$ 称为原问题 $p^*$ 的**松弛**，如果：
- 松弛问题的可行集包含原问题的可行集
- 在交集上两个问题的目标函数相同

性质：$p^*_{\text{rel}} \leq p^*$（松弛给出下界）

**凸松弛**：将非凸问题松弛为凸问题，便于求解和获得下界。

**限制（Restriction）**

定义 2.11（限制问题） 问题 $p^*_{\text{res}}$ 称为原问题的**限制**，如果：
- 限制问题的可行集是原问题可行集的子集
- 在子集上两个问题的目标函数相同

性质：$p^*_{\text{res}} \geq p^*$（限制给出上界）

### 2.3.3 非凸问题的凸松弛

**例 2.4 布尔规划**

考虑布尔规划问题：

$$\begin{align}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & Ax \leq b \\
& x_i \in \{0, 1\}, \quad i = 1, \ldots, n
\end{align}$$

这是一个非凸问题（离散约束）。凸松弛将 $x_i \in \{0, 1\}$ 松弛为 $0 \leq x_i \leq 1$：

$$\begin{align}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & Ax \leq b \\
& 0 \leq x_i \leq 1, \quad i = 1, \ldots, n
\end{align}$$

这是一个 LP，其最优值给出原问题的下界。

**例 2.5 二次规划的半定松弛**

考虑非凸 QCQP：

$$\begin{align}
\min_{x} \quad & x^T P_0 x + q_0^T x \\
\text{s.t.} \quad & x^T P_i x + q_i^T x \leq r_i, \quad i = 1, \ldots, m
\end{align}$$

其中 $P_i$ 不一定半正定。

引入 $X = xx^T$，则 $x^T P_i x = \text{tr}(P_i X)$。松弛 $X = xx^T$ 为 $X \succeq xx^T$（即 $\begin{bmatrix} X & x \\ x^T & 1 \end{bmatrix} \succeq 0$），得到 SDP 松弛：

$$\begin{align}
\min_{X, x} \quad & \text{tr}(P_0 X) + q_0^T x \\
\text{s.t.} \quad & \text{tr}(P_i X) + q_i^T x \leq r_i, \quad i = 1, \ldots, m \\
& \begin{bmatrix} X & x \\ x^T & 1 \end{bmatrix} \succeq 0
\end{align}$$

### 2.3.4 问题重构技巧

**1. 引入松弛变量**

将复杂约束转化为简单约束：

$$\min_x \|Ax - b\|_1 \Rightarrow \begin{cases} \min_{x, t} & \sum_i t_i \\ \text{s.t.} & -t_i \leq (Ax - b)_i \leq t_i \end{cases}$$

**2. 分段线性化**

对于非线性函数，可以用分段线性函数近似：

$$f(x) \approx \max_{i} (a_i^T x + b_i)$$

**3. 对偶重构**

有时求解对偶问题比原问题更容易（见第3章）。

---

## 2.4 凸优化在机器人中的应用

### 2.4.1 机器人路径规划中的凸优化

**问题描述**：给定起点 $x_{\text{start}}$ 和终点 $x_{\text{goal}}$，以及障碍物区域 $\mathcal{O}$，寻找从起点到终点的最短路径。

**凸优化建模**：

对于凸障碍物，可以将问题建模为：

$$\begin{align}
\min_{x_0, \ldots, x_T} \quad & \sum_{t=0}^{T-1} \|x_{t+1} - x_t\|_2^2 \\
\text{s.t.} \quad & x_0 = x_{\text{start}}, \quad x_T = x_{\text{goal}} \\
& x_t \in \mathcal{F}, \quad t = 0, \ldots, T
\end{align}$$

其中 $\mathcal{F} = \mathbb{R}^n \setminus \mathcal{O}$ 是自由空间。

**挑战**：自由空间 $\mathcal{F}$ 通常是非凸的（障碍物周围）。

**凸松弛方法**：

1. **凸自由空间分解**：将 $\mathcal{F}$ 分解为若干凸区域的并，在每个凸区域内求解
2. **人工势场法**：构造凸的势函数，将障碍物表示为势垒
3. **轨迹优化**：固定路径拓扑结构，优化轨迹形状（此时约束变为凸的）

### 2.4.2 机器人控制中的凸优化

**模型预测控制（MPC）**

MPC 在每个时间步求解一个开环优化问题：

$$\begin{align}
\min_{u_0, \ldots, u_{N-1}} \quad & \sum_{t=0}^{N-1} (x_t^T Q x_t + u_t^T R u_t) + x_N^T P x_N \\
\text{s.t.} \quad & x_{t+1} = A x_t + B u_t, \quad t = 0, \ldots, N-1 \\
& x_t \in \mathcal{X}, \quad u_t \in \mathcal{U}
\end{align}$$

如果 $\mathcal{X}$ 和 $\mathcal{U}$ 是凸集（如多面体、椭球），则这是一个凸优化问题（QP 或 QCQP）。

**例 2.6 机械臂的力控制**

考虑机械臂与环境接触时的力控制问题：

$$\begin{align}
\min_{\tau} \quad & \|\tau\|_2^2 + \lambda \|f - f_d\|_2^2 \\
\text{s.t.} \quad & M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau + J^T(q)f \\
& \tau_{\min} \leq \tau \leq \tau_{\max}
\end{align}$$

其中 $q$ 是关节角度，$\tau$ 是关节力矩，$f$ 是接触力，$f_d$ 是期望接触力。

对于固定的 $q, \dot{q}, \ddot{q}$，这是一个关于 $\tau$ 和 $f$ 的 QP。

### 2.4.3 机器人运动学优化

**逆运动学（IK）**

给定末端执行器位姿 $T_{\text{desired}}$，求关节角度 $q$：

$$\min_q \|f_{\text{FK}}(q) - T_{\text{desired}}\|_F^2$$

其中 $f_{\text{FK}}$ 是正运动学映射。

**挑战**：正运动学通常是非线性的（涉及三角函数）。

**凸近似方法**：

1. **迭代线性化**：在当前估计点线性化运动学，求解 QP，迭代更新
2. **凸松弛**：将旋转矩阵的 SO(3) 约束松弛为半定约束

**冗余机械臂的优化**

对于冗余机械臂（自由度大于任务空间维度），存在无穷多解。可以添加优化目标：

$$\begin{align}
\min_{q} \quad & \|q - q_{\text{nominal}}\|_2^2 \\
\text{s.t.} \quad & f_{\text{FK}}(q) = T_{\text{desired}} \\
& q_{\min} \leq q \leq q_{\max}
\end{align}$$

这是一个带等式约束的 QP（如果运动学约束线性化）。

---

## 2.5 本章小结

本章系统介绍了凸优化问题的形式、分类和转化方法：

1. **凸优化问题的定义**：目标函数凸、不等式约束凸、等式约束仿射。关键性质：局部最优即全局最优。

2. **标准问题类型**：
   - 线性规划（LP）：目标函数和约束都是仿射的
   - 二次规划（QP）：目标函数是凸二次的，约束是仿射的
   - 二次约束二次规划（QCQP）：目标和约束都是凸二次的
   - 半定规划（SDP）：涉及半正定矩阵锥
   - 锥规划：统一框架，包括 LP、SOCP、SDP

3. **问题转化**：
   - 等价变换：变量替换、目标函数变换
   - 松弛与限制：获得上下界
   - 凸松弛：将非凸问题松弛为凸问题

4. **机器人应用**：路径规划、模型预测控制、逆运动学等都可以建模为（或近似为）凸优化问题。

---

## 习题

### 基础题

2.1 判断下列问题是否是凸优化问题，并说明理由：
   (a) $\min_x x^3$ s.t. $x \geq 0$
   (b) $\min_x x^4$ s.t. $x^2 - 1 \leq 0$
   (c) $\min_{x, y} x + y$ s.t. $x^2 + y^2 = 1$
   (d) $\min_x \|Ax - b\|_2$ s.t. $\|x\|_1 \leq 1$

2.2 将下列问题转化为标准 LP 形式：
   $$\min_x \|Ax - b\|_1 \quad \text{s.t.} \quad \|x\|_\infty \leq 1$$

2.3 证明：对于严格凸二次规划（$P \succ 0$），最优解是唯一的。

2.4 将下列问题转化为 SDP：
   $$\min_x \|Ax - b\|_2^2 + \lambda \|x\|_1$$

### 提高题

2.5 考虑带等式约束的 QP：
   $$\begin{align} \min_x \quad & \frac{1}{2}x^T P x + q^T x \\ \text{s.t.} \quad & Ax = b \end{align}$$
   其中 $P \succ 0$。用拉格朗日乘子法推导最优解的解析表达式。

2.6 证明：对于 QCQP，如果 $P_0 \succ 0$ 且存在严格可行点，则强对偶性成立。

2.7 考虑布尔规划：
   $$\min_x c^T x \quad \text{s.t.} \quad Ax \leq b, \quad x_i \in \{0, 1\}$$
   推导其 SDP 松弛。

2.8 设计一个 MPC 控制器，使得双积分器系统 $\ddot{x} = u$ 从初始状态到达目标状态，同时最小化控制能量。

### 编程题

2.9 使用 CVXPY 实现并求解以下问题：
   (a) 随机生成的 LP
   (b) 最小二乘问题
   (c) 带约束的 QP

2.10 实现一个简单的轨迹优化器：
   - 给定起点、终点和障碍物
   - 使用凸优化求解平滑轨迹
   - 可视化结果

---

## 参考文献

1. Boyd S, Vandenberghe L. Convex Optimization[M]. Cambridge University Press, 2004.
2. Nocedal J, Wright S. Numerical Optimization[M]. Springer, 2006.
3. Ben-Tal A, Nemirovski A. Lectures on Modern Convex Optimization[M]. SIAM, 2001.
4. 袁亚湘, 孙文瑜. 最优化理论与方法[M]. 科学出版社, 1997.
