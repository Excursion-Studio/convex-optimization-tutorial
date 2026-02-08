# 第8章 流形的基本概念

## 8.1 拓扑流形和微分流形

### 8.1.1 流形的直观理解

流形（Manifold）是现代数学中最重要的概念之一，它是欧几里得空间的局部推广。直观上，流形是一个在每一点附近都"看起来像"欧几里得空间的拓扑空间。

**定义 8.1（拓扑流形）** 一个**n维拓扑流形** $M$ 是一个满足以下条件的拓扑空间：

1. **局部欧几里得性**：对于每一点 $p \in M$，存在 $p$ 的一个开邻域 $U$ 和一个同胚映射 $\varphi: U \to V$，其中 $V$ 是 $\mathbb{R}^n$ 中的开集。
2. **豪斯多夫性**：$M$ 是豪斯多夫空间（任意两个不同点有不相交的邻域）。
3. **第二可数性**：$M$ 具有可数拓扑基。

同胚映射 $\varphi: U \to V$ 称为**坐标卡**（Coordinate Chart），$(U, \varphi)$ 称为一个**坐标卡对**。

**例 8.1** 欧几里得空间 $\mathbb{R}^n$ 本身是一个n维拓扑流形。对于任意点 $p \in \mathbb{R}^n$，取 $U = \mathbb{R}^n$，$\varphi = \text{id}$（恒等映射）即可。

**例 8.2** n维球面 $S^n = \{(x_1, \ldots, x_{n+1}) \in \mathbb{R}^{n+1} : x_1^2 + \cdots + x_{n+1}^2 = 1\}$ 是一个n维拓扑流形。

对于 $S^2 \subset \mathbb{R}^3$，可以使用球极投影构造坐标卡：
- 北极投影：$U_N = S^2 \setminus \{(0,0,1)\}$，$\varphi_N(x,y,z) = \left(\frac{x}{1-z}, \frac{y}{1-z}\right)$
- 南极投影：$U_S = S^2 \setminus \{(0,0,-1)\}$，$\varphi_S(x,y,z) = \left(\frac{x}{1+z}, \frac{y}{1+z}\right)$

**例 8.3** 环面 $T^2 = S^1 \times S^1$ 是一个2维拓扑流形。它可以看作是 $\mathbb{R}^2$ 关于格点 $\mathbb{Z}^2$ 的商空间。

### 8.1.2 微分流形的定义

拓扑流形只要求局部同胚于欧几里得空间，而微分流形进一步要求坐标变换是光滑的，从而可以在流形上进行微积分运算。

**定义 8.2（光滑图册）** 设 $M$ 是n维拓扑流形，$\mathcal{A} = \{(U_\alpha, \varphi_\alpha)\}_{\alpha \in I}$ 是一族坐标卡，满足：
1. $\{U_\alpha\}$ 覆盖 $M$（即 $M = \bigcup_{\alpha \in I} U_\alpha$）
2. 对于任意 $\alpha, \beta \in I$，若 $U_\alpha \cap U_\beta \neq \emptyset$，则坐标变换
$$\varphi_\beta \circ \varphi_\alpha^{-1}: \varphi_\alpha(U_\alpha \cap U_\beta) \to \varphi_\beta(U_\alpha \cap U_\beta)$$
是光滑映射（$C^\infty$）。

则称 $\mathcal{A}$ 为 $M$ 的一个**光滑图册**（Smooth Atlas）。

**定义 8.3（微分流形）** 一个**n维微分流形**（或光滑流形）是一个拓扑流形 $M$ 连同其上一个极大的光滑图册。两个光滑图册称为**等价**的，如果它们的并仍然是光滑图册。

**定义 8.4（光滑函数）** 设 $M$ 是微分流形，$f: M \to \mathbb{R}$ 是一个函数。称 $f$ 在点 $p \in M$ 处**光滑**，如果存在包含 $p$ 的坐标卡 $(U, \varphi)$，使得 $f \circ \varphi^{-1}: \varphi(U) \to \mathbb{R}$ 在 $\varphi(p)$ 处光滑。

### 8.1.3 流形的维数和定向

**定义 8.5（流形的维数）** 连通微分流形 $M$ 的**维数**是唯一确定的整数 $n$，使得每一点都有邻域同胚于 $\mathbb{R}^n$。

**定义 8.6（可定向流形）** n维微分流形 $M$ 称为**可定向**的，如果存在一个光滑图册 $\mathcal{A} = \{(U_\alpha, \varphi_\alpha)\}$，使得对于任意两个坐标卡，坐标变换的雅可比行列式恒正：
$$\det\left(\frac{\partial(\varphi_\beta \circ \varphi_\alpha^{-1})^i}{\partial x^j}\right) > 0$$

**例 8.4** 
- 球面 $S^n$ 是可定向的
- 莫比乌斯带是不可定向的
- 实射影空间 $\mathbb{R}P^n$ 当 $n$ 为奇数时可定向，当 $n$ 为偶数时不可定向

### 8.1.4 机器人中的流形实例

在机器人学中，许多重要的空间都具有流形结构：

**例 8.5（特殊正交群 SO(3)）** 三维旋转矩阵的集合：
$$SO(3) = \{R \in \mathbb{R}^{3 \times 3} : R^T R = I, \det(R) = 1\}$$

$SO(3)$ 是一个3维光滑流形，可以嵌入到 $\mathbb{R}^9$ 中。

**例 8.6（特殊欧几里得群 SE(3)）** 三维刚体变换的集合：
$$SE(3) = \left\{\begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix} : R \in SO(3), t \in \mathbb{R}^3\right\}$$

$SE(3)$ 是一个6维光滑流形。

**例 8.7（机器人构型空间）** 具有 $n$ 个旋转关节的机械臂的构型空间是 $n$ 维环面 $T^n = (S^1)^n$。

## 8.2 切空间和余切空间

### 8.2.1 切向量的定义

在欧几里得空间中，切向量可以直观地理解为从一点出发的箭头。在流形上，我们需要更抽象的定义。

**定义 8.7（曲线）** 设 $M$ 是微分流形，**光滑曲线**是指光滑映射 $\gamma: (-\epsilon, \epsilon) \to M$。

**定义 8.8（切向量——几何定义）** 设 $p \in M$，考虑所有满足 $\gamma(0) = p$ 的光滑曲线 $\gamma$。两条曲线 $\gamma_1$ 和 $\gamma_2$ 称为**等价**的，如果对于任意在 $p$ 处光滑的函数 $f$，有：
$$\left.\frac{d}{dt}\right|_{t=0} f(\gamma_1(t)) = \left.\frac{d}{dt}\right|_{t=0} f(\gamma_2(t))$$

一个**切向量**就是这样的等价类 $[\gamma]$。

**定义 8.9（切向量——导子定义）** 点 $p$ 处的**切向量**是一个映射 $v: C^\infty(M) \to \mathbb{R}$，满足：
1. **线性性**：$v(af + bg) = av(f) + bv(g)$
2. **莱布尼茨法则**：$v(fg) = v(f)g(p) + f(p)v(g)$

这样的映射称为在 $p$ 处的**导子**（Derivation）。

**定义 8.10（切空间）** 点 $p$ 处所有切向量的集合记为 $T_p M$，称为 $M$ 在 $p$ 处的**切空间**。

**定理 8.1** $T_p M$ 是一个n维实向量空间，其中 $n = \dim M$。

### 8.2.2 切空间的坐标表示

设 $(U, \varphi)$ 是包含 $p$ 的坐标卡，坐标函数为 $\varphi = (x^1, \ldots, x^n)$。定义**坐标切向量**：
$$\left.\frac{\partial}{\partial x^i}\right|_p(f) = \left.\frac{\partial(f \circ \varphi^{-1})}{\partial x^i}\right|_{\varphi(p)}$$

**定理 8.2** $\left\{\left.\frac{\partial}{\partial x^i}\right|_p\right\}_{i=1}^n$ 构成 $T_p M$ 的一组基，称为**坐标基**。

任意切向量 $v \in T_p M$ 可以表示为：
$$v = \sum_{i=1}^n v^i \left.\frac{\partial}{\partial x^i}\right|_p$$

其中 $v^i = v(x^i)$ 称为 $v$ 的**分量**。

**坐标变换公式**：设 $(\tilde{x}^1, \ldots, \tilde{x}^n)$ 是另一组坐标，则：
$$\frac{\partial}{\partial \tilde{x}^j} = \sum_{i=1}^n \frac{\partial x^i}{\partial \tilde{x}^j} \frac{\partial}{\partial x^i}$$

### 8.2.3 余切空间

**定义 8.11（余切空间）** 切空间 $T_p M$ 的对偶空间 $(T_p M)^*$ 称为**余切空间**，记为 $T_p^* M$。其中的元素称为**余切向量**或**1-形式**。

**定义 8.12（微分）** 设 $f \in C^\infty(M)$，定义 $df_p \in T_p^* M$ 为：
$$df_p(v) = v(f), \quad \forall v \in T_p M$$

$df_p$ 称为 $f$ 在 $p$ 处的**微分**。

**定理 8.3** $\{dx^i|_p\}_{i=1}^n$ 构成 $T_p^* M$ 的一组基，且是 $\left\{\left.\frac{\partial}{\partial x^i}\right|_p\right\}$ 的对偶基：
$$dx^i\left(\frac{\partial}{\partial x^j}\right) = \delta^i_j$$

任意1-形式 $\omega \in T_p^* M$ 可以表示为：
$$\omega = \sum_{i=1}^n \omega_i dx^i$$

### 8.2.4 切丛和余切丛

**定义 8.13（切丛）** 流形 $M$ 的**切丛**定义为：
$$TM = \bigsqcup_{p \in M} T_p M = \{(p, v) : p \in M, v \in T_p M\}$$

切丛 $TM$ 本身是一个 $2n$ 维光滑流形。

**定义 8.14（余切丛）** 流形 $M$ 的**余切丛**定义为：
$$T^*M = \bigsqcup_{p \in M} T_p^* M = \{(p, \omega) : p \in M, \omega \in T_p^* M\}$$

余切丛 $T^*M$ 也是一个 $2n$ 维光滑流形，在哈密顿力学中具有重要作用。

## 8.3 流形上的向量场和张量场

### 8.3.1 向量场

**定义 8.15（向量场）** 流形 $M$ 上的**光滑向量场** $X$ 是一个映射，它对每一点 $p \in M$ 指定一个切向量 $X_p \in T_p M$，且满足光滑性条件：对于任意 $f \in C^\infty(M)$，函数 $X(f): M \to \mathbb{R}$ 定义为 $X(f)(p) = X_p(f)$ 是光滑的。

所有光滑向量场的集合记为 $\mathfrak{X}(M)$。

在坐标卡 $(U, \varphi)$ 中，向量场可以表示为：
$$X = \sum_{i=1}^n X^i \frac{\partial}{\partial x^i}$$

其中 $X^i: U \to \mathbb{R}$ 是光滑函数。

**定义 8.16（李括号）** 设 $X, Y \in \mathfrak{X}(M)$，定义**李括号** $[X, Y]$ 为：
$$[X, Y](f) = X(Y(f)) - Y(X(f)), \quad \forall f \in C^\infty(M)$$

**定理 8.4** 李括号满足以下性质：
1. **双线性性**：$[aX + bY, Z] = a[X, Z] + b[Y, Z]$
2. **反对称性**：$[X, Y] = -[Y, X]$
3. **雅可比恒等式**：$[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0$

因此，$(\mathfrak{X}(M), [\cdot, \cdot])$ 构成一个李代数。

### 8.3.2 张量场

**定义 8.17（张量）** 设 $V$ 是有限维实向量空间，一个 $(r, s)$ 型**张量**是一个多重线性映射：
$$T: \underbrace{V^* \times \cdots \times V^*}_{r} \times \underbrace{V \times \cdots \times V}_{s} \to \mathbb{R}$$

所有 $(r, s)$ 型张量的集合记为 $T^r_s(V)$。

**定义 8.18（张量场）** 流形 $M$ 上的**$(r, s)$ 型光滑张量场** $T$ 是一个映射，它对每一点 $p \in M$ 指定一个张量 $T_p \in T^r_s(T_p M)$，且满足光滑性条件。

在坐标卡中，张量场可以表示为：
$$T = \sum_{i_1,\ldots,i_r,j_1,\ldots,j_s} T^{i_1\cdots i_r}_{j_1\cdots j_s} \frac{\partial}{\partial x^{i_1}} \otimes \cdots \otimes \frac{\partial}{\partial x^{i_r}} \otimes dx^{j_1} \otimes \cdots \otimes dx^{j_s}$$

**例 8.8** 
- $(0, 0)$ 型张量场 = 光滑函数
- $(1, 0)$ 型张量场 = 向量场
- $(0, 1)$ 型张量场 = 1-形式场
- $(1, 1)$ 型张量场 = 线性算子场（如恒等映射对应 $\delta^i_j$）

### 8.3.3 黎曼度量

**定义 8.19（黎曼度量）** 流形 $M$ 上的**黎曼度量** $g$ 是一个 $(0, 2)$ 型光滑对称张量场，满足：
1. **对称性**：$g(X, Y) = g(Y, X)$
2. **正定性**：$g(X, X) \geq 0$，且 $g(X, X) = 0$ 当且仅当 $X = 0$

装备了黎曼度量的流形称为**黎曼流形**，记为 $(M, g)$。

在坐标卡中，黎曼度量可以表示为：
$$g = \sum_{i,j=1}^n g_{ij} dx^i \otimes dx^j$$

其中 $g_{ij} = g\left(\frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j}\right)$ 构成对称正定矩阵。

**例 8.9（欧几里得度量）** 在 $\mathbb{R}^n$ 上，标准黎曼度量为：
$$g = \sum_{i=1}^n dx^i \otimes dx^i$$

即 $g_{ij} = \delta_{ij}$。

**例 8.10（球面度量）** 在 $S^2$ 上，使用球坐标 $(\theta, \phi)$，标准度量为：
$$g = d\theta \otimes d\theta + \sin^2\theta \, d\phi \otimes d\phi$$

## 8.4 本章小结

本章介绍了流形的基本概念，包括：

1. **拓扑流形**：局部同胚于欧几里得空间的拓扑空间
2. **微分流形**：具有光滑坐标变换的流形，允许进行微积分运算
3. **切空间**：流形上某点处所有切向量的集合，是进行线性近似的空间
4. **余切空间**：切空间的对偶空间，包含1-形式
5. **向量场和张量场**：流形上的光滑切向量和张量分布
6. **黎曼度量**：定义了流形上长度和角度的结构

这些概念为后续学习流形上的优化算法奠定了数学基础。

## 8.5 参考文献

1. Lee, J. M. (2012). *Introduction to Smooth Manifolds* (2nd ed.). Springer.
2. do Carmo, M. P. (1992). *Riemannian Geometry*. Birkhäuser.
3. Absil, P. A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
4. Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press.

## 8.6 练习题

1. 证明 $S^n$ 是一个n维拓扑流形，并构造其光滑图册。

2. 设 $M$ 是微分流形，$f \in C^\infty(M)$。证明 $df$ 是 $(0, 1)$ 型张量场。

3. 在 $\mathbb{R}^3$ 中，设 $X = y\frac{\partial}{\partial z} - z\frac{\partial}{\partial y}$，$Y = z\frac{\partial}{\partial x} - x\frac{\partial}{\partial z}$。计算 $[X, Y]$。

4. 证明李括号满足雅可比恒等式。

5. 设 $(M, g)$ 是黎曼流形，在坐标卡 $(U, x^i)$ 中，证明度量矩阵 $(g_{ij})$ 是正定的。

6. 考虑 $SO(3)$ 作为 $\mathbb{R}^{3 \times 3}$ 的子流形。证明其在单位元 $I$ 处的切空间为：
   $$T_I SO(3) = \{A \in \mathbb{R}^{3 \times 3} : A + A^T = 0\}$$
   即所有 $3 \times 3$ 反对称矩阵的集合。

7. 设 $f: M \to N$ 是光滑映射，定义**微分**（或**推前**）$f_*: TM \to TN$。证明对于 $p \in M$，$f_*|_{T_p M}: T_p M \to T_{f(p)} N$ 是线性映射。

8. 在黎曼流形 $(M, g)$ 上，向量场 $X$ 的**散度**定义为 $\text{div} X = \text{tr}(\nabla X)$。在坐标卡中，证明：
   $$\text{div} X = \frac{1}{\sqrt{|g|}} \frac{\partial}{\partial x^i}(\sqrt{|g|} X^i)$$
   其中 $|g| = \det(g_{ij})$。
