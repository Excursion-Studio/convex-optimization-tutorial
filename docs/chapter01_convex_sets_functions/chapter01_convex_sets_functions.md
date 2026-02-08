# 第1章 凸集与凸函数

## 本章导读

凸集与凸函数是凸优化理论的基石。本章从几何直观出发，逐步建立严格的数学定义，为后续学习凸优化问题奠定坚实基础。通过本章学习，你将掌握凸集和凸函数的判定方法，理解共轭函数的重要性质，并学会将这些概念应用于机器人路径规划等实际问题。

---

## 1.1 凸集的定义和基本性质

### 1.1.1 凸集的几何直观与严格定义

**几何直观**

在二维平面上，一个集合是凸的，当且仅当连接集合中任意两点的线段完全包含在该集合内。直观地说，凸集没有"凹陷"或"洞"。

例如：
- 圆盘（包括边界）是凸集
- 三角形（包括内部）是凸集
- 月牙形、环形不是凸集

**严格定义**

定义 1.1（凸集） 设 $C$ 是 $\mathbb{R}^n$ 中的集合，若对任意 $x_1, x_2 \in C$ 和任意 $\theta \in [0, 1]$，都有：

$$\theta x_1 + (1-\theta) x_2 \in C$$

则称 $C$ 为**凸集**（convex set）。

表达式 $\theta x_1 + (1-\theta) x_2$ 称为 $x_1$ 和 $x_2$ 的**凸组合**（convex combination）。当 $\theta$ 在 $[0,1]$ 变化时，该表达式描述连接 $x_1$ 和 $x_2$ 的线段。

**推广定义**

更一般地，点 $x_1, x_2, \ldots, x_k$ 的凸组合定义为：

$$x = \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_k x_k$$

其中 $\theta_i \geq 0$ 对所有 $i$ 成立，且 $\sum_{i=1}^k \theta_i = 1$。

定理 1.1 集合 $C$ 是凸集当且仅当 $C$ 中任意有限个点的凸组合仍属于 $C$。

*证明*：

（必要性）设 $C$ 是凸集，用数学归纳法证明。

- 基础情形（$k=2$）：由凸集定义直接得到。
- 归纳假设：假设对 $k$ 个点成立。
- 归纳步骤：考虑 $k+1$ 个点 $x_1, \ldots, x_{k+1} \in C$ 及其凸组合：
  
  $$x = \sum_{i=1}^{k+1} \theta_i x_i = \sum_{i=1}^k \theta_i x_i + \theta_{k+1} x_{k+1}$$
  
  令 $\theta = \sum_{i=1}^k \theta_i = 1 - \theta_{k+1}$，若 $\theta > 0$，则：
  
  $$x = \theta \left(\sum_{i=1}^k \frac{\theta_i}{\theta} x_i\right) + \theta_{k+1} x_{k+1}$$
  
  由归纳假设，$y = \sum_{i=1}^k \frac{\theta_i}{\theta} x_i \in C$，再由凸集定义，$x \in C$。

（充分性）显然，取 $k=2$ 即得凸集定义。$\square$

### 1.1.2 常见凸集

**1. 超平面（Hyperplane）**

定义 1.2 给定非零向量 $a \in \mathbb{R}^n$ 和标量 $b \in \mathbb{R}$，集合：

$$H = \{x \in \mathbb{R}^n \mid a^T x = b\}$$

称为**超平面**。

超平面是 $\mathbb{R}^n$ 中的 $(n-1)$ 维仿射子空间。在 $\mathbb{R}^2$ 中，超平面是直线；在 $\mathbb{R}^3$ 中，超平面是平面。

定理 1.2 超平面是凸集。

*证明*：设 $x_1, x_2 \in H$，即 $a^T x_1 = b$，$a^T x_2 = b$。对任意 $\theta \in [0,1]$：

$$a^T(\theta x_1 + (1-\theta)x_2) = \theta a^T x_1 + (1-\theta)a^T x_2 = \theta b + (1-\theta)b = b$$

因此 $\theta x_1 + (1-\theta)x_2 \in H$。$\square$

**2. 半空间（Halfspace）**

定义 1.3 给定非零向量 $a \in \mathbb{R}^n$ 和标量 $b \in \mathbb{R}$，集合：

$$H_- = \{x \in \mathbb{R}^n \mid a^T x \leq b\}$$

称为**闭半空间**（闭的由于包含边界超平面）。

类似地，$H_+ = \{x \in \mathbb{R}^n \mid a^T x \geq b\}$ 也是闭半空间。

定理 1.3 半空间是凸集。

*证明*：设 $x_1, x_2 \in H_-$，即 $a^T x_1 \leq b$，$a^T x_2 \leq b$。对任意 $\theta \in [0,1]$：

$$a^T(\theta x_1 + (1-\theta)x_2) = \theta a^T x_1 + (1-\theta)a^T x_2 \leq \theta b + (1-\theta)b = b$$

因此 $\theta x_1 + (1-\theta)x_2 \in H_-$。$\square$

**3. 球（Ball）**

定义 1.4 以 $x_c \in \mathbb{R}^n$ 为中心，$r > 0$ 为半径的**欧几里得球**定义为：

$$B(x_c, r) = \{x \in \mathbb{R}^n \mid \|x - x_c\|_2 \leq r\}$$

$$= \{x_c + ru \mid \|u\|_2 \leq 1\}$$

定理 1.4 欧几里得球是凸集。

*证明*：设 $x_1, x_2 \in B(x_c, r)$，即 $\|x_1 - x_c\|_2 \leq r$，$\|x_2 - x_c\|_2 \leq r$。对任意 $\theta \in [0,1]$：

$$\|\theta x_1 + (1-\theta)x_2 - x_c\|_2 = \|\theta(x_1 - x_c) + (1-\theta)(x_2 - x_c)\|_2$$

$$\leq \theta\|x_1 - x_c\|_2 + (1-\theta)\|x_2 - x_c\|_2 \leq \theta r + (1-\theta)r = r$$

因此 $\theta x_1 + (1-\theta)x_2 \in B(x_c, r)$。$\square$

**4. 多面体（Polyhedron）**

定义 1.5 **多面体**是有限个半空间和超平面的交集：

$$P = \{x \in \mathbb{R}^n \mid a_i^T x \leq b_i, i = 1, \ldots, m; \quad c_j^T x = d_j, j = 1, \ldots, p\}$$

多面体可以写成更紧凑的矩阵形式：

$$P = \{x \in \mathbb{R}^n \mid Ax \preceq b, \quad Cx = d\}$$

其中 $A \in \mathbb{R}^{m \times n}$，$C \in \mathbb{R}^{p \times n}$，$\preceq$ 表示分量-wise 不等式。

定理 1.5 多面体是凸集。

*证明*：半空间和超平面都是凸集，而凸集的交集仍是凸集（见下一节），因此多面体是凸集。$\square$

**5. 单纯形（Simplex）**

定义 1.6 给定 $k+1$ 个仿射独立的点 $v_0, v_1, \ldots, v_k \in \mathbb{R}^n$，**单纯形**定义为：

$$C = \text{conv}\{v_0, v_1, \ldots, v_k\} = \left\{\sum_{i=0}^k \theta_i v_i \mid \theta_i \geq 0, \sum_{i=0}^k \theta_i = 1\right\}$$

其中 $\text{conv}$ 表示凸包（convex hull）。

单纯形是多面体的特例。常见的单纯形包括：
- 0-单纯形：点
- 1-单纯形：线段
- 2-单纯形：三角形
- 3-单纯形：四面体

**6. 锥（Cone）**

定义 1.7 集合 $K \subseteq \mathbb{R}^n$ 称为**锥**（cone），如果对任意 $x \in K$ 和 $\theta \geq 0$，有 $\theta x \in K$。

若锥 $K$ 同时也是凸集，则称为**凸锥**（convex cone）。

凸锥的等价定义：集合 $K$ 是凸锥当且仅当对任意 $x_1, x_2 \in K$ 和 $\theta_1, \theta_2 \geq 0$，有 $\theta_1 x_1 + \theta_2 x_2 \in K$。

常见的凸锥包括：

- **非负象限**：$\mathbb{R}_+^n = \{x \in \mathbb{R}^n \mid x_i \geq 0, i = 1, \ldots, n\}$
- **半正定锥**：$\mathbb{S}_+^n = \{X \in \mathbb{S}^n \mid X \succeq 0\}$，其中 $\mathbb{S}^n$ 是 $n \times n$ 对称矩阵空间
- **二阶锥**：$\mathcal{Q}^n = \{(x, t) \in \mathbb{R}^{n+1} \mid \|x\|_2 \leq t\}$

### 1.1.3 凸集的运算

**1. 交集**

定理 1.6 任意多个凸集的交集仍是凸集。

*证明*：设 $\{C_i\}_{i \in I}$ 是一族凸集，$C = \bigcap_{i \in I} C_i$。对任意 $x_1, x_2 \in C$ 和 $\theta \in [0,1]$，由于 $x_1, x_2 \in C_i$ 对所有 $i$ 成立，且每个 $C_i$ 是凸集，故 $\theta x_1 + (1-\theta)x_2 \in C_i$ 对所有 $i$ 成立。因此 $\theta x_1 + (1-\theta)x_2 \in C$。$\square$

**2. 仿射变换**

定义 1.8 映射 $f: \mathbb{R}^n \to \mathbb{R}^m$ 称为**仿射变换**，如果它具有形式：

$$f(x) = Ax + b$$

其中 $A \in \mathbb{R}^{m \times n}$，$b \in \mathbb{R}^m$。

定理 1.7 凸集在仿射变换下的像和原像都是凸集。

*证明*：

（像的凸性）设 $C$ 是凸集，$f(x) = Ax + b$。对任意 $y_1, y_2 \in f(C)$，存在 $x_1, x_2 \in C$ 使得 $y_1 = f(x_1)$，$y_2 = f(x_2)$。对任意 $\theta \in [0,1]$：

$$\theta y_1 + (1-\theta)y_2 = \theta(Ax_1 + b) + (1-\theta)(Ax_2 + b) = A(\theta x_1 + (1-\theta)x_2) + b = f(\theta x_1 + (1-\theta)x_2)$$

由于 $C$ 是凸集，$\theta x_1 + (1-\theta)x_2 \in C$，因此 $\theta y_1 + (1-\theta)y_2 \in f(C)$。

（原像的凸性）设 $D$ 是凸集，$C = f^{-1}(D) = \{x \mid f(x) \in D\}$。对任意 $x_1, x_2 \in C$ 和 $\theta \in [0,1]$：

$$f(\theta x_1 + (1-\theta)x_2) = \theta f(x_1) + (1-\theta)f(x_2) \in D$$

因此 $\theta x_1 + (1-\theta)x_2 \in C$。$\square$

**3. 透视变换**

定义 1.9 **透视函数** $P: \mathbb{R}^{n+1} \to \mathbb{R}^n$ 定义为：

$$P(x, t) = \frac{x}{t}, \quad \text{dom } P = \{(x, t) \mid t > 0\}$$

透视变换将高维空间中的点投影到低维空间。

定理 1.8 凸集在透视函数下的像和原像都是凸集。

**4. 凸包**

定义 1.10 集合 $S \subseteq \mathbb{R}^n$ 的**凸包** $\text{conv}(S)$ 是包含 $S$ 的最小凸集，即所有包含 $S$ 的凸集的交集：

$$\text{conv}(S) = \bigcap\{C \mid C \text{ 是凸集}, S \subseteq C\}$$

等价地，凸包可以表示为 $S$ 中点的所有凸组合的集合：

$$\text{conv}(S) = \left\{\sum_{i=1}^k \theta_i x_i \mid x_i \in S, \theta_i \geq 0, \sum_{i=1}^k \theta_i = 1, k \geq 1\right\}$$

定理 1.9（Carathéodory 定理） 设 $S \subseteq \mathbb{R}^n$，则 $\text{conv}(S)$ 中任意点都可以表示为 $S$ 中至多 $n+1$ 个点的凸组合。

### 1.1.4 凸集在机器人中的应用示例

**机器人构型空间的凸性分析**

在机器人运动规划中，构型空间（configuration space）描述了机器人所有可能的位姿。对于某些简单机器人，其自由构型空间（不与障碍物碰撞的构型集合）可能是凸集或凸集的并。

例 1.1 考虑平面上的二维平移机器人（只能平移，不能旋转），其构型空间为 $\mathbb{R}^2$。若障碍物是凸多边形，则自由构型空间是凸集的补集的交集，通常是非凸的。但若只有一个凸障碍物，自由构型空间可以分解为若干凸区域的并。

例 1.2 机械臂的可达工作空间（workspace）分析中，若关节具有凸约束（如位置限制），则可达工作空间可以用凸集来近似或界定。

---

## 1.2 凸函数的定义和性质

### 1.2.1 凸函数的几何直观与严格定义

**几何直观**

函数 $f$ 是凸的，当且仅当其图像上方的区域（epigraph）是凸集。直观地说，凸函数的图像"向上弯曲"，任意两点间的弦位于图像上方。

**严格定义**

定义 1.11（凸函数） 设函数 $f: \mathbb{R}^n \to \mathbb{R}$，定义域 $\text{dom } f$ 是凸集。若对任意 $x_1, x_2 \in \text{dom } f$ 和任意 $\theta \in [0, 1]$，都有：

$$f(\theta x_1 + (1-\theta) x_2) \leq \theta f(x_1) + (1-\theta) f(x_2)$$

则称 $f$ 为**凸函数**（convex function）。

若上述不等式对 $\theta \in (0, 1)$ 和 $x_1 \neq x_2$ 严格成立，则称 $f$ 为**严格凸函数**。

定义 1.12（凹函数） 若 $-f$ 是凸函数，则称 $f$ 为**凹函数**。

**一阶条件**

定理 1.10（一阶条件） 设 $f$ 可微，$\text{dom } f$ 是凸集。则 $f$ 是凸函数当且仅当：

$$f(y) \geq f(x) + \nabla f(x)^T(y - x), \quad \forall x, y \in \text{dom } f$$

*证明*：

（必要性）设 $f$ 是凸函数。对任意 $x, y \in \text{dom } f$ 和 $\theta \in (0, 1]$：

$$f(x + \theta(y - x)) \leq (1-\theta)f(x) + \theta f(y) = f(x) + \theta(f(y) - f(x))$$

因此：

$$\frac{f(x + \theta(y - x)) - f(x)}{\theta} \leq f(y) - f(x)$$

令 $\theta \to 0^+$，左边趋于方向导数 $\nabla f(x)^T(y - x)$，故：

$$\nabla f(x)^T(y - x) \leq f(y) - f(x)$$

即 $f(y) \geq f(x) + \nabla f(x)^T(y - x)$。

（充分性）设一阶条件成立。对任意 $x_1, x_2 \in \text{dom } f$ 和 $\theta \in [0,1]$，令 $x = \theta x_1 + (1-\theta)x_2$。由一阶条件：

$$f(x_1) \geq f(x) + \nabla f(x)^T(x_1 - x) = f(x) + (1-\theta)\nabla f(x)^T(x_1 - x_2)$$

$$f(x_2) \geq f(x) + \nabla f(x)^T(x_2 - x) = f(x) - \theta\nabla f(x)^T(x_1 - x_2)$$

第一式乘以 $\theta$，第二式乘以 $(1-\theta)$，相加得：

$$\theta f(x_1) + (1-\theta)f(x_2) \geq f(x) = f(\theta x_1 + (1-\theta)x_2)$$

因此 $f$ 是凸函数。$\square$

几何解释：凸函数的一阶条件表明，函数图像上任意点的切线（切平面）位于图像下方。

**二阶条件**

定理 1.11（二阶条件） 设 $f$ 二阶可微，$\text{dom } f$ 是凸集。则：

- $f$ 是凸函数当且仅当Hessian矩阵半正定：$\nabla^2 f(x) \succeq 0$，$\forall x \in \text{dom } f$
- 若 $\nabla^2 f(x) \succ 0$（正定），则 $f$ 是严格凸函数

*证明概要*：

利用泰勒展开：

$$f(y) = f(x) + \nabla f(x)^T(y-x) + \frac{1}{2}(y-x)^T\nabla^2 f(z)(y-x)$$

其中 $z$ 在 $x$ 和 $y$ 之间。若 $\nabla^2 f \succeq 0$，则：

$$f(y) \geq f(x) + \nabla f(x)^T(y-x)$$

满足一阶条件，故 $f$ 是凸函数。

反之，若 $f$ 是凸函数，假设存在 $x$ 使得 $\nabla^2 f(x)$ 不是半正定，则存在方向 $v$ 使得 $v^T\nabla^2 f(x)v < 0$。对足够小的 $t$，令 $y = x + tv$，泰勒展开与凸性矛盾。$\square$

### 1.2.2 常见凸函数

**1. 线性函数和仿射函数**

定理 1.12 线性函数 $f(x) = a^T x$ 和仿射函数 $f(x) = a^T x + b$ 既是凸函数也是凹函数。

*证明*：$\nabla^2 f(x) = 0$，既半正定也半负定。$\square$

**2. 二次函数**

定理 1.13 二次函数 $f(x) = \frac{1}{2}x^T P x + q^T x + r$（其中 $P \in \mathbb{S}^n$）是凸函数当且仅当 $P \succeq 0$。

*证明*：$\nabla^2 f(x) = P$，由二阶条件即得。$\square$

**3. 指数函数和对数函数**

定理 1.14 
- $e^{ax}$ 在 $\mathbb{R}$ 上是凸函数（$a \in \mathbb{R}$）
- $\log x$ 在 $\mathbb{R}_{++}$ 上是凹函数
- $x\log x$ 在 $\mathbb{R}_{++}$ 上是凸函数

**4. 范数**

定理 1.15 任意范数 $\|\cdot\|$ 是凸函数。

*证明*：由范数的三角不等式和齐次性：

$$\|\theta x + (1-\theta)y\| \leq \|\theta x\| + \|(1-\theta)y\| = \theta\|x\| + (1-\theta)\|y\|$$

$\square$

常见的范数包括：
- $l_1$ 范数：$\|x\|_1 = \sum_{i=1}^n |x_i|$
- $l_2$ 范数（欧几里得范数）：$\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$
- $l_\infty$ 范数：$\|x\|_\infty = \max_i |x_i|$

**5. 最大值函数**

定理 1.16 $f(x) = \max\{x_1, x_2, \ldots, x_n\}$ 是凸函数。

*证明*：对任意 $x, y \in \mathbb{R}^n$ 和 $\theta \in [0,1]$：

$$f(\theta x + (1-\theta)y) = \max_i(\theta x_i + (1-\theta)y_i) \leq \max_i(\theta x_i) + \max_i((1-\theta)y_i) = \theta f(x) + (1-\theta)f(y)$$

$\square$

**6. 对数行列式**

定理 1.17 $f(X) = \log\det X$ 在正定矩阵空间 $\mathbb{S}_{++}^n$ 上是凹函数。

### 1.2.3 凸函数的运算

**1. 非负加权和**

定理 1.18 若 $f_1, f_2, \ldots, f_m$ 是凸函数，$w_i \geq 0$，则 $f = \sum_{i=1}^m w_i f_i$ 是凸函数。

*证明*：直接由定义验证。$\square$

**2. 与仿射变换的复合**

定理 1.19 若 $f$ 是凸函数，$g(x) = f(Ax + b)$，则 $g$ 是凸函数。

*证明*：设 $x_1, x_2 \in \text{dom } g$ 和 $\theta \in [0,1]$：

$$g(\theta x_1 + (1-\theta)x_2) = f(A(\theta x_1 + (1-\theta)x_2) + b) = f(\theta(Ax_1 + b) + (1-\theta)(Ax_2 + b))$$

$$\leq \theta f(Ax_1 + b) + (1-\theta)f(Ax_2 + b) = \theta g(x_1) + (1-\theta)g(x_2)$$

$\square$

**3. 逐点最大值**

定理 1.20 若 $f_1, f_2, \ldots, f_m$ 是凸函数，则 $f(x) = \max\{f_1(x), f_2(x), \ldots, f_m(x)\}$ 是凸函数。

*证明*：$f$ 的上图是各 $f_i$ 上图的交集，凸集的交集是凸集。$\square$

**4. 复合函数**

定理 1.21 设 $h: \mathbb{R} \to \mathbb{R}$ 是凸且非减函数，$g: \mathbb{R}^n \to \mathbb{R}$ 是凸函数，则 $f(x) = h(g(x))$ 是凸函数。

*证明*：利用链式法则和凸性定义可证。$\square$

### 1.2.4 水平集和上镜图

定义 1.13 函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的**上镜图**（epigraph）定义为：

$$\text{epi } f = \{(x, t) \in \mathbb{R}^{n+1} \mid x \in \text{dom } f, f(x) \leq t\}$$

定理 1.22 函数 $f$ 是凸函数当且仅当其上镜图 $\text{epi } f$ 是凸集。

定义 1.14 函数 $f$ 的**水平集**（level set）定义为：

$$C_\alpha = \{x \in \text{dom } f \mid f(x) \leq \alpha\}$$

定理 1.23 若 $f$ 是凸函数，则其所有水平集都是凸集。

*证明*：设 $x_1, x_2 \in C_\alpha$，即 $f(x_1) \leq \alpha$，$f(x_2) \leq \alpha$。对任意 $\theta \in [0,1]$：

$$f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta)f(x_2) \leq \theta \alpha + (1-\theta)\alpha = \alpha$$

因此 $\theta x_1 + (1-\theta)x_2 \in C_\alpha$。$\square$

注意：水平集是凸集的函数称为**拟凸函数**（quasiconvex function），拟凸函数不一定是凸函数。

---

## 1.3 共轭函数

### 1.3.1 共轭函数的定义

共轭函数是凸分析中的重要工具，它将函数转换为对偶空间中的表示。

定义 1.15（共轭函数） 函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的**共轭函数** $f^*: \mathbb{R}^n \to \mathbb{R}$ 定义为：

$$f^*(y) = \sup_{x \in \text{dom } f} (y^T x - f(x))$$

几何解释：$f^*(y)$ 是线性函数 $y^T x$ 与 $f(x)$ 之间的最大"间隙"。

### 1.3.2 共轭函数的性质

定理 1.24 共轭函数 $f^*$ 是闭凸函数（无论 $f$ 是否是凸函数）。

*证明*：$f^*$ 是一族仿射函数（关于 $y$）的上确界，仿射函数既是凸的也是闭的，凸闭函数的上确界仍是凸闭的。$\square$

定理 1.25（Fenchel 不等式） 对任意 $x, y \in \mathbb{R}^n$：

$$f(x) + f^*(y) \geq x^T y$$

*证明*：由共轭函数的定义，$f^*(y) \geq y^T x - f(x)$ 对所有 $x$ 成立，移项即得。$\square$

定理 1.26（二次共轭） 若 $f$ 是闭凸函数，则 $f^{**} = f$。

### 1.3.3 共轭函数的计算示例

**例 1.3 指示函数的共轭**

设 $C$ 是凸集，指示函数定义为：

$$I_C(x) = \begin{cases} 0 & x \in C \\ +\infty & x \notin C \end{cases}$$

其共轭函数为：

$$I_C^*(y) = \sup_{x \in C} y^T x$$

这称为集合 $C$ 的**支撑函数**（support function）。

**例 1.4 范数的共轭**

设 $f(x) = \|x\|$ 是任意范数，其共轭函数为：

$$f^*(y) = \begin{cases} 0 & \|y\|_* \leq 1 \\ +\infty & \|y\|_* > 1 \end{cases}$$

其中 $\|\cdot\|_*$ 是对偶范数。

对于 $l_2$ 范数，由于 $l_2$ 范数是自对偶的，故：

$$f^*(y) = \begin{cases} 0 & \|y\|_2 \leq 1 \\ +\infty & \|y\|_2 > 1 \end{cases}$$

**例 1.5 二次函数的共轭**

设 $f(x) = \frac{1}{2}x^T Q x$，其中 $Q \succ 0$。求 $f^*(y)$。

解：对固定的 $y$，最大化 $y^T x - \frac{1}{2}x^T Q x$。令梯度为零：

$$y - Qx = 0 \Rightarrow x = Q^{-1}y$$

代入得：

$$f^*(y) = y^T Q^{-1}y - \frac{1}{2}(Q^{-1}y)^T Q (Q^{-1}y) = \frac{1}{2}y^T Q^{-1}y$$

### 1.3.4 共轭函数在优化中的应用

共轭函数在优化中有重要应用，特别是在对偶问题的构造中。

考虑优化问题：

$$\min_x f(x) + g(Ax)$$

利用共轭函数，可以构造其对偶问题：

$$\max_y -f^*(A^T y) - g^*(-y)$$

这种对偶形式在分布式优化和机器学习中有广泛应用。

---

## 1.4 本章小结

本章系统介绍了凸集和凸函数的基本理论：

1. **凸集**：满足任意两点连线仍在集合内的集合。常见凸集包括超平面、半空间、球、多面体等。凸集在交集、仿射变换下保持封闭。

2. **凸函数**：满足 Jensen 不等式的函数。判定方法包括定义法、一阶条件（切线在图像下方）和二阶条件（Hessian 半正定）。

3. **共轭函数**：通过上确界定义的变换，将函数映射到对偶空间。共轭函数总是闭凸的，且二次共轭恢复原函数（对闭凸函数）。

这些概念为后续学习凸优化问题、对偶理论和算法设计奠定了数学基础。

---

## 习题

### 基础题

1.1 证明以下集合是凸集：
   (a) 半正定锥 $\mathbb{S}_+^n = \{X \in \mathbb{S}^n \mid X \succeq 0\}$
   (b) 二阶锥 $\mathcal{Q}^n = \{(x, t) \in \mathbb{R}^{n+1} \mid \|x\|_2 \leq t\}$

1.2 判断下列函数是否是凸函数，并说明理由：
   (a) $f(x) = e^x$ 在 $\mathbb{R}$ 上
   (b) $f(x) = x^3$ 在 $\mathbb{R}$ 上
   (c) $f(x) = \|Ax - b\|_2^2$
   (d) $f(x) = \max\{x, x^2\}$ 在 $\mathbb{R}$ 上

1.3 设 $f$ 是凸函数，$g$ 是凸且非减函数。证明 $h(x) = g(f(x))$ 是凸函数。

1.4 计算下列函数的共轭函数：
   (a) $f(x) = |x|$ 在 $\mathbb{R}$ 上
   (b) $f(x) = e^x$ 在 $\mathbb{R}$ 上

### 提高题

1.5 证明：若 $C$ 是凸集，则其内部 $\text{int}(C)$ 和闭包 $\overline{C}$ 也是凸集。

1.6 设 $f$ 是凸函数，$\text{dom } f$ 是开集。证明 $f$ 在 $\text{dom } f$ 上连续。

1.7 证明 Jensen 不等式：若 $f$ 是凸函数，$x_1, \ldots, x_k \in \text{dom } f$，$\theta_1, \ldots, \theta_k \geq 0$ 且 $\sum \theta_i = 1$，则：

$$f\left(\sum_{i=1}^k \theta_i x_i\right) \leq \sum_{i=1}^k \theta_i f(x_i)$$

1.8 设 $f$ 是可微凸函数，证明：$x^*$ 是 $f$ 的全局最小值点当且仅当 $\nabla f(x^*) = 0$。

### 编程题

1.9 编写 Python 程序可视化以下凸集：
   (a) 二维空间中的 $l_1$ 球、$l_2$ 球和 $l_\infty$ 球
   (b) 由不等式 $Ax \leq b$ 定义的多面体

1.10 实现梯度下降法求解二次优化问题 $\min_x \frac{1}{2}x^T P x + q^T x$，其中 $P \succ 0$。比较不同步长选择策略的收敛速度。

---

## 参考文献

1. Boyd S, Vandenberghe L. Convex Optimization[M]. Cambridge University Press, 2004.
2. Rockafellar R T. Convex Analysis[M]. Princeton University Press, 1970.
3. Bertsekas D P. Convex Optimization Theory[M]. Athena Scientific, 2009.
4. 袁亚湘, 孙文瑜. 最优化理论与方法[M]. 科学出版社, 1997.
