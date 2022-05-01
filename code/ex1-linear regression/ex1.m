%% 
%% 2 Linear regression with one variable
%% 2.1 plotting the data
%%
% ex1data1.txt
data = load('./ex1data1.txt');
x = data(:, 1); % 人口
y = data(:, 2); % 利润
m = length(y);
plot(x, y, 'rx', 'markerSize', 10);
ylabel('profit in $10,000s');
xlabel('population of city in 10,000s');
%% 2.2 gradient decent
%% 2.2.1 update equations
% $$j(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2\\h_\theta(x) 
% = \theta^Tx = \theta_01 + \theta_1x_1\\\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) 
% - y^{(i)})x^{(i)}_j$$
% 
% 注意 $x^{\left(j\right)}$ 是对应 $\theta_j$ 的特征数据，这里已经统一用向量思维了
%% 2.2.2 implementation
% 增加一个维度给 x，用于对应 $\theta_0$

X = x; % 保留
x = [ones(m, 1), data(:, 1)];
x

theta = zeros(2, 1);
theta
iterations = 1500;
alpha = 0.01;
%% 2.2.3 computing the cost $j\left(\theta \right)$

cost_func(x, y, theta, m)
%% 2.2.4 gradient descent
% we minimize the value of _J_(_θ_) by changing the values of the vector _θ_, 
% not by changing _X _or _y_  

x
y
theta = [0.11; 0.14]
theta = zeros(2, 1)
h     = x * theta
cost  = sum(x .* (h - y), 1);
cost.'
% 理解以上的形状后，谢自定义函数 gd
train_theta = gd(x, y, 1000, 0.01)
x * train_theta
%% 
% *注意这里 X 和 x 的使用，*X 是一维，x 加入了一个维度

figure;
plot(X, y, 'rx', X, x * train_theta, 'b', 'MarkerSize', 10);
%% 2.4 *Visualizing *$J\left(\theta \right)$


%% 3. *Linear regression with multiple variables*
%% 3.1 *Feature Normalization*
%%
data = load('./ex1data2.txt');
%% 
% the size of the house,  the number of bedrooms, the price of the house

data
%% 
% 发现房屋面积远大于房间数，这样不利于收敛( 这个可以详见李宏毅老师用等高线图的解释

m = mean(data, 1); % 沿着第一维度(行)，不断累加，所以第一维度消失，

%% 
% 
% 
% 
% 
% 
%% 函数自定义区
%%
function cost = cost_func(x, y, theta, m)
    % x .* y 对应元素相乘
    h = x * theta;
    c = (h - y) .^ 2;
    cost = sum(c) / (2 * m);
end


%% 
% $$j(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2\\h_\theta(x) 
% = \theta^Tx = \theta_01 + \theta_1x_1\\\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) 
% - y^{(i)})x^{(i)}_j$$
% 
% 多注意一下矩阵形状。

% 梯度下降，拟合出参数 
function theta = gd(x, y, iterations, alpha)
    theta = zeros(2, 1);
    m = length(y);
    
    for i = 1 : iterations
        h    = x * theta;
        cost = sum((h - y) .* x, 1); % 沿着列加和，这是第一维度
        theta = theta - alpha * cost.' ./ m; % 不转置会发生广播
    end
end
%% 
% 
% 
%