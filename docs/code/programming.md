# 编程练习

## 练习1：梯度下降法实现

**任务**：实现梯度下降法，求解函数 $f(x) = x_1^2 + 2x_2^2$ 的最小值。

**参考**：梯度下降法在教程 [第4章 无约束优化算法](chapter04/chapter4_unconstrained_optimization.md) 的 4.1.1 节中有详细介绍。

**步骤**：

1. 定义函数和梯度计算函数。
2. 实现梯度下降算法。
3. 测试不同的初始点和步长。

**Python代码示例**：

```python
import numpy as np

def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

def gradient_descent(grad_f, x0, alpha, max_iter=1000, tol=1e-6):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        x = x - alpha * grad
    return x

# 测试
x0 = np.array([2, 1])
alpha = 0.1
x_opt = gradient_descent(grad_f, x0, alpha)
print("最优解:", x_opt)
print("最优值:", f(x_opt))
```

**Matlab代码示例**：

```matlab
function [f_val] = f(x)
    f_val = x(1)^2 + 2*x(2)^2;
end

function [grad] = grad_f(x)
    grad = [2*x(1); 4*x(2)];
end

function [x_opt] = gradient_descent(grad_f, x0, alpha, max_iter, tol)
    if nargin < 4
        max_iter = 1000;
    end
    if nargin < 5
        tol = 1e-6;
    end
    
    x = x0;
    for i = 1:max_iter
        grad = feval(grad_f, x);
        if norm(grad) < tol
            break;
        end
        x = x - alpha * grad;
    end
    x_opt = x;
end

% 测试
x0 = [2; 1];
alpha = 0.1;
x_opt = gradient_descent(@grad_f, x0, alpha);
f_opt = f(x_opt);
disp(['最优解: ', num2str(x_opt')]);
disp(['最优值: ', num2str(f_opt)]);
```

**梯度下降法在深度学习中的应用**：

梯度下降法是深度学习中最基础的优化算法，被广泛应用于神经网络的训练。在深度学习中，梯度下降法有以下变体：

1. **随机梯度下降（SGD）**：每次只使用一个样本计算梯度，适合大规模数据集。
2. **小批量梯度下降（Mini-batch SGD）**：每次使用一小批样本计算梯度，平衡了计算效率和梯度估计的准确性。
3. **动量法（Momentum）**：引入动量项，加速在平坦区域的收敛。
4. **Adam**：结合了动量法和自适应学习率，是目前深度学习中最常用的优化算法之一。

这些变体通过改进基本的梯度下降法，使得神经网络的训练更加高效和稳定。

## 练习2：线性回归实现

**任务**：使用最小二乘法实现线性回归，拟合给定的数据。

**参考**：线性回归在教程 [第7章 应用案例](chapter07/chapter7_applications.md) 的 7.1.1 节中有详细介绍。

**步骤**：

1. 准备数据。
2. 构造设计矩阵。
3. 使用最小二乘法求解参数。
4. 绘制拟合结果。

**Python代码示例**：

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
```

**Matlab代码示例**：

```matlab
% 准备数据
x = [1; 2; 3; 4; 5];
y = [2; 4; 5; 7; 8];

% 构造设计矩阵
X = [ones(length(x), 1), x];

% 最小二乘法求解
w = inv(X' * X) * X' * y;

% 预测
x_pred = linspace(0, 6, 100);
y_pred = w(1) + w(2) * x_pred;

% 绘制结果
figure;
scatter(x, y, 'b', 'filled', 'DisplayName', '数据点');
hold on;
plot(x_pred, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', '拟合直线');
xlabel('x');
ylabel('y');
legend('Location', 'best');
title('线性回归拟合结果');
hold off;

% 显示参数
disp(['参数: ', num2str(w')]);
```

**线性回归在机器学习中的应用**：

线性回归是机器学习中最基础的监督学习算法之一，虽然简单，但在许多领域仍有广泛应用：

1. **作为基准模型**：在尝试复杂模型之前，线性回归通常被用作基准，帮助了解数据的基本趋势。
2. **特征重要性分析**：通过线性回归的系数大小，可以评估各个特征对预测目标的影响程度。
3. **正则化变体**：岭回归（Ridge）、LASSO和弹性网络（Elastic Net）等正则化方法，在处理高维数据和多重共线性问题时表现优异。
4. **集成学习的基础**：许多集成学习算法（如随机森林、梯度提升树）在构建过程中会使用线性模型作为基学习器。
5. **时间序列预测**：线性回归可以扩展为自回归模型（AR）和自回归综合移动平均模型（ARIMA），用于时间序列数据的预测。

在线性回归的基础上，深度学习通过增加网络层数和非线性激活函数，能够捕捉更复杂的数据模式，解决线性模型无法处理的非线性问题。

## 练习3：使用CVXPY求解线性规划问题

**任务**：使用CVXPY求解以下线性规划问题：

**参考**：线性规划在教程 [第3章 凸优化问题](chapter03/chapter3_convex_optimization_problems.md) 的 3.3.1 节中有详细介绍。

$$\max_{x} 3x_1 + 4x_2$$

$$\text{s.t. } x_1 + x_2 \leq 5$$

$$2x_1 + x_2 \leq 8$$

$$x_1, x_2 \geq 0$$

**Python代码示例**：

```python
import cvxpy as cp

# 创建变量
x = cp.Variable(2, nonneg=True)

# 定义目标函数
objective = cp.Maximize(3*x[0] + 4*x[1])

# 定义约束
constraints = [
    x[0] + x[1] <= 5,
    2*x[0] + x[1] <= 8
]

# 创建问题并求解
problem = cp.Problem(objective, constraints)
problem.solve()

# 打印结果
print("状态:", problem.status)
print("最优解:", x.value)
print("最优值:", problem.value)
```

**Matlab代码示例**：

```matlab
% 安装CVX工具箱后使用
% 可从 http://cvxr.com/cvx/ 下载安装

cvx_begin
    variables x(2) % 定义变量
    maximize(3*x(1) + 4*x(2)) % 定义目标函数
    subject to % 定义约束
        x(1) + x(2) <= 5;
        2*x(1) + x(2) <= 8;
        x >= 0; % 变量非负
cvx_end

% 打印结果
disp(['状态: ', cvx_status]);
disp(['最优解: ', num2str(x')]);
disp(['最优值: ', num2str(cvx_optval)]);
```

**线性规划在供应链管理中的应用**：

线性规划是运筹学的重要分支，在供应链管理中有着广泛的应用：

1. **生产计划优化**：确定不同产品的生产数量，以最小化生产成本或最大化利润，同时满足资源约束和需求约束。
2. **库存管理**：优化库存水平，平衡持有成本和缺货成本，确保供应链的顺畅运行。
3. **运输路线优化**：确定从多个仓库到多个需求点的最优运输路线，最小化运输成本。
4. **资源分配**：在有限资源（如人力、设备、原材料）的情况下，最优分配资源以实现目标最大化。
5. **供应链网络设计**：设计最优的供应链网络结构，包括工厂、仓库的选址和容量规划。

随着供应链的复杂性增加，线性规划模型也在不断扩展，与其他方法（如模拟、机器学习）相结合，以应对更复杂的实际问题。

## 练习4：使用拉格朗日乘数法求解约束优化问题

**任务**：使用拉格朗日乘数法求解下列问题：

**参考**：拉格朗日乘数法在教程 [第5章 约束优化算法](chapter05/chapter5_constrained_optimization.md) 的 5.1.1 节中有详细介绍。

$$\min_{x} x_1^2 + x_2^2$$

$$\text{s.t. } x_1 + x_2 = 1$$

**Python代码示例**：

```python
import numpy as np
from scipy.optimize import minimize

# 定义目标函数
def objective(x):
    return x[0]**2 + x[1]**2

# 定义约束条件
def constraint(x):
    return x[0] + x[1] - 1

# 初始猜测
x0 = np.array([0.5, 0.5])

# 定义约束
cons = [{'type': 'eq', 'fun': constraint}]

# 求解
result = minimize(objective, x0, constraints=cons, method='SLSQP')

print("最优解:", result.x)
print("最优值:", result.fun)
```

**Matlab代码示例**：

```matlab
% 方法1：使用fmincon函数求解

% 定义目标函数
fun = @(x) x(1)^2 + x(2)^2;

% 初始猜测
x0 = [0.5; 0.5];

% 定义约束
A = [];
b = [];
Aeq = [1, 1];
beq = 1;
lb = [];
ub = [];

% 求解
options = optimoptions('fmincon', 'Display', 'iter');
[x_opt, f_opt] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, [], options);

% 打印结果
disp(['最优解: ', num2str(x_opt')]);
disp(['最优值: ', num2str(f_opt)]);

% 方法2：解析解法（拉格朗日乘数法）
% 构造拉格朗日函数：L = x1² + x2² - λ(x1 + x2 - 1)
% 求导并令导数为零：
% ∂L/∂x1 = 2x1 - λ = 0
% ∂L/∂x2 = 2x2 - λ = 0
% ∂L/∂λ = -(x1 + x2 - 1) = 0
% 解得：x1 = x2 = 1/2, λ = 1

fprintf('\n解析解法结果：\n');
disp(['最优解: [0.5, 0.5]']);
disp(['最优值: 0.5']);
```

**拉格朗日乘数法在约束优化中的应用**：

拉格朗日乘数法是求解约束优化问题的经典方法，虽然在处理复杂问题时计算量较大，但它为理解约束优化的本质提供了重要 insights：

1. **理论基础**：拉格朗日乘数法是KKT条件的基础，而KKT条件是约束优化问题的一阶最优性条件。
2. **几何意义**：拉格朗日条件的几何意义是在最优解处，目标函数的梯度是约束函数梯度的线性组合。
3. **扩展应用**：
   - **对偶理论**：拉格朗日乘数法是对偶理论的基础，通过求解对偶问题可以获得原问题的下界。
   - **内点法**：现代内点法通过引入障碍函数，将约束优化问题转化为一系列无约束优化问题，其本质仍与拉格朗日乘数法相关。
   - **机器学习**：在支持向量机（SVM）中，拉格朗日乘数法被用于求解最大间隔分类器。
   - **经济学**：拉格朗日乘数在经济学中被解释为影子价格，表示约束资源的边际价值。

在实际应用中，对于复杂的约束优化问题，通常使用数值方法（如内点法、SQP等）来求解，但拉格朗日乘数法的思想仍然贯穿其中。

## 练习5：实现岭回归

**任务**：实现岭回归，拟合给定的数据。

**参考**：岭回归在教程 [第7章 应用案例](chapter07/chapter7_applications.md) 的 7.1.1 节中有详细介绍。

**Python代码示例**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 7, 8])

# 添加噪声
y_noisy = y + np.random.normal(0, 0.5, len(y))

# 构造设计矩阵
X = np.vstack([np.ones(len(x)), x]).T

# 岭回归参数
lambdas = [0, 0.1, 1, 10]

plt.scatter(x, y_noisy, label='带噪声数据')

for lambd in lambdas:
    # 岭回归求解
    w = np.linalg.inv(X.T @ X + lambd * np.eye(2)) @ X.T @ y_noisy
    
    # 预测
    x_pred = np.linspace(0, 6, 100)
    y_pred = w[0] + w[1] * x_pred
    
    plt.plot(x_pred, y_pred, label=f'λ={lambd}')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("参数:", w)
```

**Matlab代码示例**：

```matlab
% 准备数据
x = [1; 2; 3; 4; 5];
y = [2; 4; 5; 7; 8];

% 添加噪声
rng(42); % 设置随机种子
y_noisy = y + randn(length(y), 1) * 0.5;

% 构造设计矩阵
X = [ones(length(x), 1), x];

% 岭回归参数
lambdas = [0, 0.1, 1, 10];

% 绘制结果
figure;
scatter(x, y_noisy, 'b', 'filled', 'DisplayName', '带噪声数据');
hold on;

% 求解不同λ值的岭回归
colors = {'r', 'g', 'm', 'c'};
for i = 1:length(lambdas)
    lambda_val = lambdas(i);
    % 岭回归求解
    w = inv(X' * X + lambda_val * eye(2)) * X' * y_noisy;
    
    % 预测
    x_pred = linspace(0, 6, 100);
    y_pred = w(1) + w(2) * x_pred;
    
    % 绘制拟合直线
    plot(x_pred, y_pred, colors{i}, 'LineWidth', 2, 'DisplayName', ['λ=', num2str(lambda_val)]);
end

xlabel('x');
ylabel('y');
legend('Location', 'best');
title('岭回归不同正则化参数的拟合结果');
grid on;
hold off;

% 显示最后一个λ值的参数
disp(['参数: ', num2str(w')]);
```

**岭回归在机器学习中的应用**：

岭回归作为线性回归的正则化变体，在机器学习和数据科学中有广泛的应用：

1. **处理多重共线性**：当输入特征之间存在高度相关性时，岭回归通过L2正则化可以稳定模型参数，避免过拟合。

2. **高维数据处理**：在特征维度大于样本数量的情况下，岭回归可以提供稳定的模型参数估计。

3. **模型选择**：通过交叉验证选择最优的正则化参数λ，可以平衡模型的复杂度和泛化能力。

4. **与深度学习的联系**：
   - **权重衰减**：深度学习中的权重衰减技术本质上就是岭回归的L2正则化，用于防止神经网络过拟合。
   - **贝叶斯视角**：岭回归可以看作是在参数上施加高斯先验的贝叶斯线性回归，这与深度学习中常用的高斯过程和变分推断有密切联系。
   - **迁移学习**：在迁移学习中，岭回归可以用于微调预训练模型的参数，通过正则化来保留原始模型的知识。

5. **实际应用场景**：
   - **金融预测**：在金融时间序列预测中，岭回归可以处理高度相关的经济指标。
   - **图像处理**：在图像处理中，岭回归可以用于特征提取和降维。
   - **生物信息学**：在基因表达数据分析中，岭回归可以处理高维的基因表达数据。

## 练习6：使用牛顿法求解无约束优化问题

**任务**：使用牛顿法求解函数 $f(x) = x_1^2 + 2x_2^2$ 的最小值。

**参考**：牛顿法在教程 [第4章 无约束优化算法](chapter04/chapter4_unconstrained_optimization.md) 的 4.2.1 节中有详细介绍。

**Python代码示例**：

```python
import numpy as np

def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

def hess_f(x):
    return np.array([[2, 0], [0, 4]])

def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        hess = hess_f(x)
        # 求解线性方程组: H * d = -g
        d = np.linalg.solve(hess, -grad)
        x = x + d
    return x

# 测试
x0 = np.array([2, 1])
x_opt = newton_method(f, grad_f, hess_f, x0)
print("最优解:", x_opt)
print("最优值:", f(x_opt))
```

**Matlab代码示例**：

```matlab
% 定义函数
function [f_val] = f(x)
    f_val = x(1)^2 + 2*x(2)^2;
end

% 定义梯度
function [grad] = grad_f(x)
    grad = [2*x(1); 4*x(2)];
end

% 定义海森矩阵
function [hess] = hess_f(x)
    hess = [2, 0; 0, 4];
end

% 牛顿法实现
function [x_opt] = newton_method(f, grad_f, hess_f, x0, max_iter, tol)
    if nargin < 5
        max_iter = 100;
    end
    if nargin < 6
        tol = 1e-6;
    end
    
    x = x0;
    for i = 1:max_iter
        grad = feval(grad_f, x);
        if norm(grad) < tol
            break;
        end
        hess = feval(hess_f, x);
        % 求解线性方程组: H * d = -g
        d = hess \ (-grad);
        x = x + d;
    end
    x_opt = x;
end

% 测试
x0 = [2; 1];
x_opt = newton_method(@f, @grad_f, @hess_f, x0);
f_opt = f(x_opt);

% 打印结果
disp(['最优解: ', num2str(x_opt')]);
disp(['最优值: ', num2str(f_opt)]);

% 显示迭代过程
fprintf('\n迭代过程演示：\n');
x_current = x0;
fprintf('初始点: [%f, %f], 函数值: %f\n', x_current(1), x_current(2), f(x_current));

% 手动执行一次迭代
grad = grad_f(x_current);
hess = hess_f(x_current);
d = hess \ (-grad);
x_next = x_current + d;
fprintf('一次迭代后: [%f, %f], 函数值: %f\n', x_next(1), x_next(2), f(x_next));
fprintf('梯度范数: %f\n', norm(grad_f(x_next)));
```

**牛顿法在优化中的应用**：

牛顿法作为一种二阶优化算法，在许多领域都有重要应用：

1. **数学优化**：牛顿法是求解无约束优化问题的经典算法，特别适用于二次可微函数。

2. **统计学习**：
   - **最大似然估计**：牛顿法被用于求解最大似然估计问题，特别是在逻辑回归和广义线性模型中。
   - **Fisher评分法**：牛顿法的一个变体，专门用于统计参数估计，利用Fisher信息矩阵作为海森矩阵的估计。

3. **与深度学习的联系**：
   - **二阶优化方法**：虽然纯牛顿法在深度学习中由于计算海森矩阵的高复杂度而不常用，但其变体如K-FAC（Kronecker-Factored Approximate Curvature）和Shampoo等被用于大规模神经网络训练。
   - **曲率信息**：牛顿法利用海森矩阵提供的曲率信息，这启发了许多自适应学习率算法（如Adam）的设计。
   - **模型优化**：在小批量设置下，牛顿法的思想被用于改进神经网络的训练效率。

4. **强化学习**：
   - **策略优化**：在策略梯度方法中，牛顿法被用于优化策略参数，提高收敛速度。
   - **自然策略梯度**：利用Fisher信息矩阵作为度量，实现更有效的策略更新，这与牛顿法的思想密切相关。

5. **实际应用场景**：
   - **金融建模**：牛顿法被用于求解金融衍生品定价模型中的非线性方程。
   - **工程优化**：在工程设计和参数优化中，牛顿法被用于快速找到最优解。
   - **数值分析**：牛顿法是求解非线性方程组的有效方法，被广泛应用于科学计算。

尽管牛顿法在计算复杂度上高于一阶方法，但其快速的收敛速度使其在许多需要高精度解的场景中仍然是首选方法。

## 练习7：使用共轭梯度法求解无约束优化问题

**任务**：使用共轭梯度法求解函数 $f(x) = x_1^2 + 2x_2^2$ 的最小值。

**参考**：共轭梯度法在教程 [第4章 无约束优化算法](chapter04/chapter4_unconstrained_optimization.md) 的 4.2.3 节中有详细介绍。

**步骤**：

1. 定义函数、梯度计算函数。
2. 实现共轭梯度算法。
3. 测试算法的收敛性。

**Python代码示例**：

```python
import numpy as np

def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

def conjugate_gradient_method(f, grad_f, x0, max_iter=100, tol=1e-6):
    x = x0
    grad = grad_f(x)
    d = -grad
    
    for i in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        
        # 计算步长 alpha
        # 对于二次函数，有解析解
        # 这里直接使用闭式解
        if i == 0:
            # 初始步长
            alpha = np.dot(grad, grad) / np.dot(d, np.array([2*d[0], 4*d[1]]))
        else:
            # 对于二次函数，共轭梯度法一步收敛
            alpha = np.dot(grad, grad) / np.dot(d, np.array([2*d[0], 4*d[1]]))
        
        # 更新 x
        x = x + alpha * d
        
        # 计算新的梯度
        new_grad = grad_f(x)
        
        # 计算 beta
        beta = np.dot(new_grad, new_grad) / np.dot(grad, grad)
        
        # 更新搜索方向
        d = -new_grad + beta * d
        
        # 更新梯度
        grad = new_grad
    
    return x

# 测试
x0 = np.array([2, 1])
x_opt = conjugate_gradient_method(f, grad_f, x0)
print("最优解:", x_opt)
print("最优值:", f(x_opt))
```

**Matlab代码示例**：

```matlab
% 定义函数
function [f_val] = f(x)
    f_val = x(1)^2 + 2*x(2)^2;
end

% 定义梯度
function [grad] = grad_f(x)
    grad = [2*x(1); 4*x(2)];
end

% 共轭梯度法实现
function [x_opt] = conjugate_gradient_method(f, grad_f, x0, max_iter, tol)
    if nargin < 5
        max_iter = 100;
    end
    if nargin < 6
        tol = 1e-6;
    end
    
    x = x0;
    grad = feval(grad_f, x);
    d = -grad;
    
    for i = 1:max_iter
        if norm(grad) < tol
            break;
        end
        
        % 计算步长 alpha
        % 对于二次函数，有解析解
        alpha = (grad' * grad) / (d' * [2*d(1); 4*d(2)]);
        
        % 更新 x
        x = x + alpha * d;
        
        % 计算新的梯度
        new_grad = feval(grad_f, x);
        
        % 计算 beta
        beta = (new_grad' * new_grad) / (grad' * grad);
        
        % 更新搜索方向
        d = -new_grad + beta * d;
        
        % 更新梯度
        grad = new_grad;
    end
    x_opt = x;
end

% 测试
x0 = [2; 1];
x_opt = conjugate_gradient_method(@f, @grad_f, x0);
f_opt = f(x_opt);

% 打印结果
disp(['最优解: ', num2str(x_opt')]);
disp(['最优值: ', num2str(f_opt)]);

% 显示迭代过程
fprintf('\n迭代过程演示：\n');
x_current = x0;
grad_current = grad_f(x_current);
d_current = -grad_current;
fprintf('初始点: [%f, %f], 函数值: %f, 梯度范数: %f\n', ...
    x_current(1), x_current(2), f(x_current), norm(grad_current));

% 执行一次迭代
alpha = (grad_current' * grad_current) / (d_current' * [2*d_current(1); 4*d_current(2)]);
x_next = x_current + alpha * d_current;
grad_next = grad_f(x_next);
beta = (grad_next' * grad_next) / (grad_current' * grad_current);
d_next = -grad_next + beta * d_current;
fprintf('一次迭代后: [%f, %f], 函数值: %f, 梯度范数: %f\n', ...
    x_next(1), x_next(2), f(x_next), norm(grad_next));
```

**共轭梯度法在优化中的应用**：

共轭梯度法是一种结合了梯度下降法和牛顿法优点的优化算法，在许多领域都有广泛应用：

1. **大规模优化问题**：共轭梯度法不需要存储海森矩阵，内存占用低，适合求解大规模优化问题。

2. **线性方程组求解**：共轭梯度法是求解大型稀疏对称正定线性方程组的首选方法之一。

3. **机器学习**：
   - **神经网络训练**：共轭梯度法被用于训练神经网络，特别是在内存受限的情况下。
   - **支持向量机**：共轭梯度法被用于求解SVM的对偶问题，尤其是在处理大规模数据集时。
   - **主成分分析**：共轭梯度法被用于求解特征值问题，是PCA的核心算法之一。

4. **科学计算**：
   - **有限元分析**：共轭梯度法被用于求解有限元方法产生的大型线性方程组。
   - **分子动力学**：共轭梯度法被用于能量最小化和分子结构优化。

5. **与现代优化方法的结合**：
   - **预条件共轭梯度法**：通过引入预条件器来加速收敛，被广泛应用于科学计算和工程模拟。
   - **随机共轭梯度法**：结合随机梯度下降的思想，用于大规模机器学习问题。

共轭梯度法的核心优势在于它对于二次函数能够在有限步内收敛，同时对于非二次函数也有较好的收敛性能，是一种非常实用的优化算法。

## 练习8：实现逻辑回归

**任务**：实现逻辑回归算法，用于二分类问题。

**参考**：逻辑回归在教程 [第7章 应用案例](chapter07/chapter7_applications.md) 的 7.1.2 节中有详细介绍。

**步骤**：

1. 准备二分类数据集。
2. 实现sigmoid函数。
3. 实现对数损失函数。
4. 实现梯度下降法求解逻辑回归参数。
5. 测试模型的分类性能。

**Python代码示例**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成二分类数据集
def generate_data(n_samples=100):
    # 类别0
    np.random.seed(42)
    class0 = np.random.normal(loc=[-1, -1], scale=0.5, size=(n_samples//2, 2))
    # 类别1
    class1 = np.random.normal(loc=[1, 1], scale=0.5, size=(n_samples//2, 2))
    # 合并数据
    X = np.vstack([class0, class1])
    # 生成标签
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    return X, y

# 实现sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 实现对数损失函数
def log_loss(y_true, y_pred):
    # 避免除零错误
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 实现逻辑回归
def logistic_regression(X, y, learning_rate=0.01, max_iter=1000, tol=1e-6):
    # 添加偏置项
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    # 初始化参数
    theta = np.zeros(X_bias.shape[1])
    
    for i in range(max_iter):
        # 计算线性组合
        z = np.dot(X_bias, theta)
        # 计算预测值
        y_pred = sigmoid(z)
        # 计算梯度
        gradient = np.dot(X_bias.T, (y_pred - y)) / len(y)
        # 计算损失
        loss = log_loss(y, y_pred)
        # 更新参数
        theta -= learning_rate * gradient
        # 检查收敛
        if np.linalg.norm(gradient) < tol:
            break
    
    return theta

# 预测函数
def predict(X, theta, threshold=0.5):
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    z = np.dot(X_bias, theta)
    y_pred = sigmoid(z)
    return (y_pred >= threshold).astype(int)

# 计算准确率
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# 测试
X, y = generate_data()
theta = logistic_regression(X, y)
y_pred = predict(X, theta)
acc = accuracy(y, y_pred)
print(f"准确率: {acc:.4f}")
print(f"模型参数: {theta}")

# 可视化决策边界
plt.figure(figsize=(10, 6))
# 绘制数据点
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='类别0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='类别1')
# 绘制决策边界
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
grid = np.c_[xx1.ravel(), xx2.ravel()]
grid_pred = predict(grid, theta)
grid_pred = grid_pred.reshape(xx1.shape)
plt.contourf(xx1, xx2, grid_pred, alpha=0.2, cmap='coolwarm')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('逻辑回归决策边界')
plt.legend()
plt.show()
```

**Matlab代码示例**：

```matlab
% 生成二分类数据集
function [X, y] = generate_data(n_samples)
    if nargin < 1
        n_samples = 100;
    end
    
    % 类别0
    rng(42); % 设置随机种子
    class0 = mvnrnd([-1, -1], 0.5*eye(2), n_samples/2);
    % 类别1
    class1 = mvnrnd([1, 1], 0.5*eye(2), n_samples/2);
    % 合并数据
    X = [class0; class1];
    % 生成标签
    y = [zeros(n_samples/2, 1); ones(n_samples/2, 1)];
end

% 实现sigmoid函数
function [z] = sigmoid(x)
    z = 1 ./ (1 + exp(-x));
end

% 实现对数损失函数
function [loss] = log_loss(y_true, y_pred)
    % 避免除零错误
    epsilon = 1e-15;
    y_pred = max(epsilon, min(1 - epsilon, y_pred));
    loss = -mean(y_true .* log(y_pred) + (1 - y_true) .* log(1 - y_pred));
end

% 实现逻辑回归
function [theta] = logistic_regression(X, y, learning_rate, max_iter, tol)
    if nargin < 3
        learning_rate = 0.01;
    end
    if nargin < 4
        max_iter = 1000;
    end
    if nargin < 5
        tol = 1e-6;
    end
    
    % 添加偏置项
    X_bias = [ones(size(X, 1), 1), X];
    % 初始化参数
    theta = zeros(size(X_bias, 2), 1);
    
    for i = 1:max_iter
        % 计算线性组合
        z = X_bias * theta;
        % 计算预测值
        y_pred = sigmoid(z);
        % 计算梯度
        gradient = X_bias' * (y_pred - y) / length(y);
        % 计算损失
        loss = log_loss(y, y_pred);
        % 更新参数
        theta = theta - learning_rate * gradient;
        % 检查收敛
        if norm(gradient) < tol
            break;
        end
    end
end

% 预测函数
function [y_pred] = predict(X, theta, threshold)
    if nargin < 3
        threshold = 0.5;
    end
    X_bias = [ones(size(X, 1), 1), X];
    z = X_bias * theta;
    y_pred_prob = sigmoid(z);
    y_pred = (y_pred_prob >= threshold);
end

% 计算准确率
function [acc] = accuracy(y_true, y_pred)
    acc = mean(y_true == y_pred);
end

% 测试
[X, y] = generate_data();
theta = logistic_regression(X, y);
y_pred = predict(X, theta);
acc = accuracy(y, y_pred);
fprintf('准确率: %.4f\n', acc);
fprintf('模型参数: ');
fprintf('%.4f ', theta);
fprintf('\n');

% 可视化决策边界
figure('Position', [100, 100, 800, 600]);
% 绘制数据点
hold on;
scatter(X(y == 0, 1), X(y == 0, 2), 'b', 'filled', 'DisplayName', '类别0');
scatter(X(y == 1, 1), X(y == 1, 2), 'r', 'filled', 'DisplayName', '类别1');
% 绘制决策边界
x1_min = min(X(:, 1)) - 0.5;
x1_max = max(X(:, 1)) + 0.5;
x2_min = min(X(:, 2)) - 0.5;
x2_max = max(X(:, 2)) + 0.5;
[xx1, xx2] = meshgrid(linspace(x1_min, x1_max, 100), linspace(x2_min, x2_max, 100));
grid = [xx1(:), xx2(:)];
grid_pred = predict(grid, theta);
grid_pred = reshape(grid_pred, size(xx1));
contourf(xx1, xx2, grid_pred, 'Alpha', 0.2, 'DisplayName', '决策边界');
xlabel('特征1');
ylabel('特征2');
title('逻辑回归决策边界');
legend('Location', 'best');
grid on;
hold off;
```

**逻辑回归在机器学习中的应用**：

逻辑回归是机器学习中最基础、最常用的分类算法之一，虽然简单，但应用广泛：

1. **二分类问题**：逻辑回归是解决二分类问题的首选算法之一，如垃圾邮件检测、欺诈检测、疾病诊断等。

2. **概率预测**：逻辑回归不仅可以给出分类结果，还可以输出样本属于某个类别的概率，这对于需要风险评估的场景非常重要。

3. **特征重要性分析**：通过逻辑回归的系数大小和符号，可以评估各个特征对分类结果的影响程度和方向。

4. **多分类扩展**：通过一对多（One-vs-Rest）或一对一（One-vs-One）策略，逻辑回归可以扩展为多分类算法。

5. **与深度学习的联系**：
   - **神经网络的基础**：逻辑回归可以看作是只有一个输出单元的单层神经网络，是神经网络的基础。
   - **激活函数**：sigmoid函数是最早的神经网络激活函数之一，虽然现在被ReLU等函数替代，但其思想仍然重要。
   - **损失函数**：逻辑回归的对数损失函数是分类任务中常用的损失函数，被广泛应用于深度学习中。

6. **实际应用场景**：
   - **金融领域**：信用评分、违约预测、欺诈检测。
   - **医疗领域**：疾病诊断、风险预测、治疗效果评估。
   - **营销领域**：客户 churn 预测、广告点击率预测、客户细分。
   - **自然语言处理**：情感分析、垃圾邮件检测、文本分类。

逻辑回归的成功之处在于它将线性模型的简单性与概率预测的能力相结合，同时保持了凸优化的特性，使得模型训练稳定且高效。

## 练习9：实现随机梯度下降（SGD）

**任务**：实现随机梯度下降算法，用于求解线性回归问题。

**参考**：随机梯度下降在教程 [第4章 无约束优化算法](chapter04/chapter4_unconstrained_optimization.md) 的 4.1.4 节中有详细介绍。

**步骤**：

1. 准备回归数据集。
2. 实现批量梯度下降（BGD）算法。
3. 实现随机梯度下降（SGD）算法。
4. 比较两种算法的收敛速度。

**Python代码示例**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成回归数据集
def generate_regression_data(n_samples=1000, noise=0.1):
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * noise
    return X, y

# 批量梯度下降（BGD）
def batch_gradient_descent(X, y, learning_rate=0.01, max_iter=1000, tol=1e-6):
    # 添加偏置项
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    # 初始化参数
    theta = np.zeros((X_bias.shape[1], 1))
    # 记录损失
    loss_history = []
    
    for i in range(max_iter):
        # 计算预测值
        y_pred = X_bias.dot(theta)
        # 计算误差
        error = y_pred - y
        # 计算梯度
        gradient = X_bias.T.dot(error) / len(y)
        # 计算损失
        loss = np.mean(error ** 2) / 2
        loss_history.append(loss)
        # 更新参数
        theta -= learning_rate * gradient
        # 检查收敛
        if np.linalg.norm(gradient) < tol:
            break
    
    return theta, loss_history

# 随机梯度下降（SGD）
def stochastic_gradient_descent(X, y, learning_rate=0.01, max_iter=1000, tol=1e-6, batch_size=1):
    # 添加偏置项
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    # 初始化参数
    theta = np.zeros((X_bias.shape[1], 1))
    # 记录损失
    loss_history = []
    n_samples = len(y)
    
    for i in range(max_iter):
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X_bias[indices]
        y_shuffled = y[indices]
        
        for j in range(0, n_samples, batch_size):
            # 取小批量数据
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]
            # 计算预测值
            y_pred = X_batch.dot(theta)
            # 计算误差
            error = y_pred - y_batch
            # 计算梯度
            gradient = X_batch.T.dot(error) / len(y_batch)
            # 更新参数
            theta -= learning_rate * gradient
        
        # 计算整个数据集的损失
        y_pred_full = X_bias.dot(theta)
        loss = np.mean((y_pred_full - y) ** 2) / 2
        loss_history.append(loss)
        
        # 检查收敛
        if loss < tol:
            break
    
    return theta, loss_history

# 测试
X, y = generate_regression_data()

# 使用BGD
theta_bgd, loss_bgd = batch_gradient_descent(X, y)

# 使用SGD
theta_sgd, loss_sgd = stochastic_gradient_descent(X, y, batch_size=32)

print("BGD参数:", theta_bgd.flatten())
print("SGD参数:", theta_sgd.flatten())

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_bgd, label='Batch Gradient Descent')
plt.plot(loss_sgd, label='Stochastic Gradient Descent (batch_size=32)')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('BGD vs SGD 收敛速度比较')
plt.legend()
plt.grid(True)
plt.show()
```

**Matlab代码示例**：

```matlab
% 生成回归数据集
function [X, y] = generate_regression_data(n_samples, noise)
    if nargin < 1
        n_samples = 1000;
    end
    if nargin < 2
        noise = 0.1;
    end
    
    rng(42); % 设置随机种子
    X = rand(n_samples, 1) * 10;
    y = 2 * X + 1 + randn(n_samples, 1) * noise;
end

% 批量梯度下降（BGD）
function [theta, loss_history] = batch_gradient_descent(X, y, learning_rate, max_iter, tol)
    if nargin < 3
        learning_rate = 0.01;
    end
    if nargin < 4
        max_iter = 1000;
    end
    if nargin < 5
        tol = 1e-6;
    end
    
    % 添加偏置项
    X_bias = [ones(size(X, 1), 1), X];
    % 初始化参数
    theta = zeros(size(X_bias, 2), 1);
    % 记录损失
    loss_history = zeros(max_iter, 1);
    
    for i = 1:max_iter
        % 计算预测值
        y_pred = X_bias * theta;
        % 计算误差
        error = y_pred - y;
        % 计算梯度
        gradient = X_bias' * error / length(y);
        % 计算损失
        loss = mean(error .^ 2) / 2;
        loss_history(i) = loss;
        % 更新参数
        theta = theta - learning_rate * gradient;
        % 检查收敛
        if norm(gradient) < tol
            loss_history = loss_history(1:i);
            break;
        end
    end
end

% 随机梯度下降（SGD）
function [theta, loss_history] = stochastic_gradient_descent(X, y, learning_rate, max_iter, tol, batch_size)
    if nargin < 3
        learning_rate = 0.01;
    end
    if nargin < 4
        max_iter = 1000;
    end
    if nargin < 5
        tol = 1e-6;
    end
    if nargin < 6
        batch_size = 1;
    end
    
    % 添加偏置项
    X_bias = [ones(size(X, 1), 1), X];
    % 初始化参数
    theta = zeros(size(X_bias, 2), 1);
    % 记录损失
    loss_history = zeros(max_iter, 1);
    n_samples = length(y);
    
    for i = 1:max_iter
        % 随机打乱数据
        indices = randperm(n_samples);
        X_shuffled = X_bias(indices, :);
        y_shuffled = y(indices);
        
        for j = 1:batch_size:n_samples
            % 取小批量数据
            end_idx = min(j+batch_size-1, n_samples);
            X_batch = X_shuffled(j:end_idx, :);
            y_batch = y_shuffled(j:end_idx);
            % 计算预测值
            y_pred = X_batch * theta;
            % 计算误差
            error = y_pred - y_batch;
            % 计算梯度
            gradient = X_batch' * error / length(y_batch);
            % 更新参数
            theta = theta - learning_rate * gradient;
        end
        
        % 计算整个数据集的损失
        y_pred_full = X_bias * theta;
        loss = mean((y_pred_full - y) .^ 2) / 2;
        loss_history(i) = loss;
        
        % 检查收敛
        if loss < tol
            loss_history = loss_history(1:i);
            break;
        end
    end
end

% 测试
[X, y] = generate_regression_data();

% 使用BGD
[theta_bgd, loss_bgd] = batch_gradient_descent(X, y);

% 使用SGD
[theta_sgd, loss_sgd] = stochastic_gradient_descent(X, y, 0.01, 1000, 1e-6, 32);

fprintf('BGD参数: %.4f %.4f\n', theta_bgd);
fprintf('SGD参数: %.4f %.4f\n', theta_sgd);

% 绘制损失曲线
figure('Position', [100, 100, 800, 600]);
plot(loss_bgd, 'b-', 'LineWidth', 2, 'DisplayName', 'Batch Gradient Descent');
hold on;
plot(loss_sgd, 'r-', 'LineWidth', 2, 'DisplayName', 'Stochastic Gradient Descent (batch_size=32)');
xlabel('迭代次数');
ylabel('损失');
title('BGD vs SGD 收敛速度比较');
legend('Location', 'best');
grid on;
hold off;
```

**随机梯度下降在深度学习中的应用**：

随机梯度下降（SGD）是深度学习中最基础、最常用的优化算法，它的变体如小批量梯度下降（Mini-batch SGD）被广泛应用于神经网络训练：

1. **大规模数据处理**：SGD通过每次只处理一个或几个样本，大大减少了内存需求，使得训练大规模数据集成为可能。

2. **噪声带来的好处**：SGD的随机性可以帮助模型跳出局部最优解，找到更好的全局最优解。

3. **深度学习中的变体**：
   - **小批量梯度下降（Mini-batch SGD）**：每次使用一小批样本计算梯度，平衡了计算效率和梯度估计的准确性。
   - **动量法（Momentum）**：引入动量项，加速在平坦区域的收敛，抑制振荡。
   - **Adagrad**：自适应学习率，对低频特征给予较大的学习率，高频特征给予较小的学习率。
   - **RMSprop**：改进的Adagrad，通过指数移动平均来调整学习率。
   - **Adam**：结合了动量法和RMSprop的优点，是目前深度学习中最常用的优化算法之一。

4. **实际应用场景**：
   - **图像分类**：如ResNet、EfficientNet等模型的训练。
   - **自然语言处理**：如BERT、GPT等预训练模型的训练。
   - **强化学习**：如DQN、PPO等算法中的策略更新。
   - **生成模型**：如GAN、VAE等模型的训练。

5. **超参数调优**：SGD的性能很大程度上依赖于学习率的选择，通常需要通过交叉验证来确定最佳学习率和批量大小。

随机梯度下降的思想不仅限于深度学习，它已经成为现代机器学习和人工智能领域中最核心的优化技术之一。

## 练习10：实现Adam优化器

**任务**：实现Adam优化器，用于求解无约束优化问题。

**介绍**：Adam（Adaptive Moment Estimation）是一种自适应学习率的优化算法，由Diederik P. Kingma和Jimmy Ba于2014年提出。它结合了动量法和RMSprop的优点，通过计算参数的一阶矩估计（动量）和二阶矩估计（自适应学习率），为每个参数动态调整学习率，从而实现更高效的收敛。

### Adam优化器的原理

Adam优化器的核心思想是：
1. **动量（Momentum）**：利用梯度的指数移动平均来加速收敛，特别是在参数空间的平坦区域，有助于减少振荡。
2. **自适应学习率**：利用梯度平方的指数移动平均来为每个参数动态调整学习率，使得不同参数可以有不同的学习率。
3. **偏差修正**：由于在训练初期，移动平均的估计值会偏向于0，Adam通过偏差修正来解决这个问题。

### Adam优化器的公式

Adam优化器的更新规则如下：

1. **计算梯度**：
   $$g_t = \nabla f(\theta_{t-1})$$

2. **更新一阶矩估计（动量）**：
   $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

3. **更新二阶矩估计（自适应学习率）**：
   $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

4. **偏差修正**：
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
   $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

5. **参数更新**：
   $$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

其中：
- $\beta_1$：一阶矩估计的指数衰减率（通常取0.9）
- $\beta_2$：二阶矩估计的指数衰减率（通常取0.999）
- $\alpha$：学习率（通常取0.001）
- $\epsilon$：防止除零错误的小常数（通常取1e-8）

### Adam优化器的特点

1. **自适应学习率**：为每个参数维护一个自适应学习率，根据参数的历史梯度信息进行调整，使得稀疏参数可以获得更大的学习率。

2. **动量项**：通过一阶矩估计（指数移动平均）来加速收敛，特别是在参数空间的平坦区域，有助于减少振荡。

3. **偏差修正**：对一阶矩和二阶矩的估计进行偏差修正，使得在训练初期也能有较好的性能。

4. **鲁棒性**：对学习率的选择不那么敏感，通常使用默认值（learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8）就能取得较好的效果。

5. **计算效率**：虽然比标准SGD复杂，但计算开销仍然相对较低，适合大规模深度学习模型的训练。

### Adam优化器的应用

Adam优化器在深度学习中被广泛应用，特别适合以下场景：
- **大规模神经网络训练**：由于其自适应学习率的特性，Adam可以有效地训练深层神经网络。
- **非平稳目标函数**：当目标函数随着时间变化时，Adam的自适应学习率能够更好地适应。
- **稀疏梯度**：对于稀疏数据，Adam可以为不同参数设置不同的学习率，提高训练效率。
- **超参数调优**：由于对学习率不那么敏感，减少了超参数调优的工作量。

### Adam优化器的变体

基于Adam的成功，研究人员提出了多种变体：
- **AdamW**：在Adam的基础上加入了权重衰减（weight decay），提高了泛化能力。
- **AMSGrad**：改进了二阶矩估计的更新方式，解决了Adam可能出现的收敛问题。
- **RAdam**：结合了SGD和Adam的优点，在训练初期使用SGD的思想，后期使用Adam。
- **Adamax**：使用无穷范数代替L2范数来计算二阶矩估计，增强了稳定性。

**步骤**：

1. 定义目标函数和梯度计算函数。
2. 实现Adam优化算法。
3. 与普通梯度下降法比较收敛速度。

**Python代码示例**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return x[0]**2 + 2*x[1]**2 + np.sin(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])

# 定义梯度计算函数
def grad_f(x):
    return np.array([
        2*x[0] + 2*np.pi*np.cos(2*np.pi*x[0]),
        4*x[1] - 2*np.pi*np.sin(2*np.pi*x[1])
    ])

# 普通梯度下降法
def gradient_descent(grad_f, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    x = x0
    x_history = [x.copy()]
    loss_history = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        x = x - learning_rate * grad
        x_history.append(x.copy())
        loss_history.append(f(x))
    
    return x, x_history, loss_history

# Adam优化器
def adam_optimizer(grad_f, x0, learning_rate=0.001, max_iter=1000, tol=1e-6, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    m = np.zeros_like(x)  # 一阶矩估计
    v = np.zeros_like(x)  # 二阶矩估计
    t = 0  # 时间步
    x_history = [x.copy()]
    loss_history = [f(x)]
    
    for i in range(max_iter):
        t += 1
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        # 更新一阶矩估计
        m = beta1 * m + (1 - beta1) * grad
        # 更新二阶矩估计
        v = beta2 * v + (1 - beta2) * grad**2
        # 偏差修正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # 参数更新
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        x_history.append(x.copy())
        loss_history.append(f(x))
    
    return x, x_history, loss_history

# 测试
x0 = np.array([1.5, 1.5])

# 使用普通梯度下降
theta_gd, x_history_gd, loss_history_gd = gradient_descent(grad_f, x0, learning_rate=0.01)

# 使用Adam优化器
theta_adam, x_history_adam, loss_history_adam = adam_optimizer(grad_f, x0, learning_rate=0.01)

print("普通梯度下降最优解:", theta_gd)
print("Adam最优解:", theta_adam)
print("普通梯度下降最优值:", f(theta_gd))
print("Adam最优值:", f(theta_adam))

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_history_gd, label='Gradient Descent')
plt.plot(loss_history_adam, label='Adam Optimizer')
plt.xlabel('迭代次数')
plt.ylabel('函数值')
plt.title('Gradient Descent vs Adam 收敛速度比较')
plt.legend()
plt.grid(True)
plt.show()

# 绘制优化路径
x_history_gd = np.array(x_history_gd)
x_history_adam = np.array(x_history_adam)

plt.figure(figsize=(10, 6))
# 绘制函数轮廓
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + 2*X2**2 + np.sin(2*np.pi*X1) + np.cos(2*np.pi*X2)
plt.contour(X1, X2, Z, levels=20, cmap='viridis')
# 绘制优化路径
plt.plot(x_history_gd[:, 0], x_history_gd[:, 1], 'o-', label='Gradient Descent')
plt.plot(x_history_adam[:, 0], x_history_adam[:, 1], 's-', label='Adam Optimizer')
plt.plot(x0[0], x0[1], 'rx', markersize=10, label='初始点')
plt.plot(theta_gd[0], theta_gd[1], 'go', markersize=8, label='GD最优解')
plt.plot(theta_adam[0], theta_adam[1], 'bo', markersize=8, label='Adam最优解')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('优化路径比较')
plt.legend()
plt.grid(True)
plt.show()
```

**Matlab代码示例**：

```matlab
% 定义目标函数
function [f_val] = f(x)
    f_val = x(1)^2 + 2*x(2)^2 + sin(2*pi*x(1)) + cos(2*pi*x(2));
end

% 定义梯度计算函数
function [grad] = grad_f(x)
    grad = [
        2*x(1) + 2*pi*cos(2*pi*x(1));
        4*x(2) - 2*pi*sin(2*pi*x(2))
    ];
end

% 普通梯度下降法
function [x_opt, x_history, loss_history] = gradient_descent(grad_f, x0, learning_rate, max_iter, tol)
    if nargin < 3
        learning_rate = 0.01;
    end
    if nargin < 4
        max_iter = 1000;
    end
    if nargin < 5
        tol = 1e-6;
    end
    
    x = x0;
    x_history = zeros(max_iter+1, length(x0));
    x_history(1, :) = x';
    loss_history = zeros(max_iter+1, 1);
    loss_history(1) = f(x);
    
    for i = 1:max_iter
        grad = feval(grad_f, x);
        if norm(grad) < tol
            x_history = x_history(1:i+1, :);
            loss_history = loss_history(1:i+1);
            break;
        end
        x = x - learning_rate * grad;
        x_history(i+1, :) = x';
        loss_history(i+1) = f(x);
    end
    
    x_opt = x;
end

% Adam优化器
function [x_opt, x_history, loss_history] = adam_optimizer(grad_f, x0, learning_rate, max_iter, tol, beta1, beta2, epsilon)
    if nargin < 3
        learning_rate = 0.001;
    end
    if nargin < 4
        max_iter = 1000;
    end
    if nargin < 5
        tol = 1e-6;
    end
    if nargin < 6
        beta1 = 0.9;
    end
    if nargin < 7
        beta2 = 0.999;
    end
    if nargin < 8
        epsilon = 1e-8;
    end
    
    x = x0;
    m = zeros(size(x));  % 一阶矩估计
    v = zeros(size(x));  % 二阶矩估计
    t = 0;  % 时间步
    x_history = zeros(max_iter+1, length(x0));
    x_history(1, :) = x';
    loss_history = zeros(max_iter+1, 1);
    loss_history(1) = f(x);
    
    for i = 1:max_iter
        t = t + 1;
        grad = feval(grad_f, x);
        
        if norm(grad) < tol
            x_history = x_history(1:i+1, :);
            loss_history = loss_history(1:i+1);
            break;
        end
        
        % 更新一阶矩估计
        m = beta1 * m + (1 - beta1) * grad;
        % 更新二阶矩估计
        v = beta2 * v + (1 - beta2) * grad.^2;
        % 偏差修正
        m_hat = m / (1 - beta1^t);
        v_hat = v / (1 - beta2^t);
        % 参数更新
        x = x - learning_rate * m_hat ./ (sqrt(v_hat) + epsilon);
        
        x_history(i+1, :) = x';
        loss_history(i+1) = f(x);
    end
    
    x_opt = x;
end

% 测试
x0 = [1.5; 1.5];

% 使用普通梯度下降
[theta_gd, x_history_gd, loss_history_gd] = gradient_descent(@grad_f, x0, 0.01);

% 使用Adam优化器
[theta_adam, x_history_adam, loss_history_adam] = adam_optimizer(@grad_f, x0, 0.01);

fprintf('普通梯度下降最优解: %.4f %.4f\n', theta_gd);
fprintf('Adam最优解: %.4f %.4f\n', theta_adam);
fprintf('普通梯度下降最优值: %.4f\n', f(theta_gd));
fprintf('Adam最优值: %.4f\n', f(theta_adam));

% 绘制损失曲线
figure('Position', [100, 100, 800, 600]);
plot(loss_history_gd, 'b-', 'LineWidth', 2, 'DisplayName', 'Gradient Descent');
hold on;
plot(loss_history_adam, 'r-', 'LineWidth', 2, 'DisplayName', 'Adam Optimizer');
xlabel('迭代次数');
ylabel('函数值');
title('Gradient Descent vs Adam 收敛速度比较');
legend('Location', 'best');
grid on;
hold off;

% 绘制优化路径
figure('Position', [100, 100, 800, 600]);
% 绘制函数轮廓
x1 = linspace(-2, 2, 100);
x2 = linspace(-2, 2, 100);
[X1, X2] = meshgrid(x1, x2);
Z = X1.^2 + 2*X2.^2 + sin(2*pi*X1) + cos(2*pi*X2);
contour(X1, X2, Z, 20, 'LineWidth', 1.5);
hold on;
% 绘制优化路径
plot(x_history_gd(:, 1), x_history_gd(:, 2), 'bo-', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'Gradient Descent');
plot(x_history_adam(:, 1), x_history_adam(:, 2), 'rs-', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'Adam Optimizer');
plot(x0(1), x0(2), 'kx', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', '初始点');
plot(theta_gd(1), theta_gd(2), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'GD最优解');
plot(theta_adam(1), theta_adam(2), 'mo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Adam最优解');
xlabel('x1');
ylabel('x2');
title('优化路径比较');
legend('Location', 'best');
grid on;
hold off;
```

**Adam优化器在深度学习中的应用**：

Adam优化器是目前深度学习中最流行的优化算法之一，它结合了动量法和自适应学习率的优点，具有以下特点：

1. **自适应学习率**：Adam为每个参数维护一个自适应学习率，根据参数的历史梯度信息来调整学习率，使得不同参数可以有不同的学习率。

2. **动量项**：通过一阶矩估计（指数移动平均）来加速收敛，特别是在参数空间的平坦区域。

3. **偏差修正**：对一阶矩和二阶矩的估计进行偏差修正，使得在训练初期也能有较好的性能。

4. **广泛适用性**：Adam在各种深度学习任务中都表现出色，包括：
   - **图像分类**：如在ImageNet数据集上训练ResNet、ViT等模型。
   - **目标检测**：如训练YOLO、Faster R-CNN等模型。
   - **自然语言处理**：如训练BERT、GPT等预训练语言模型。
   - **语音识别**：如训练语音识别模型。
   - **强化学习**：如训练DQN、PPO等强化学习算法。

5. **超参数设置**：Adam的默认超参数（learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8）在大多数情况下都能取得较好的效果，减少了超参数调优的工作量。

6. **变体**：基于Adam的变体如AdamW（加入权重衰减）、AMSGrad（改进二阶矩估计）等，进一步提高了优化性能。

7. **优缺点**：
   - **优点**：收敛速度快，对学习率不敏感，适用于大规模数据和参数的场景。
   - **缺点**：在某些任务上可能不如SGD with momentum最终收敛到更好的结果，有时会出现泛化能力差的问题。

Adam优化器的成功之处在于它平衡了收敛速度和收敛质量，为深度学习模型的训练提供了一种高效、稳定的优化方法。