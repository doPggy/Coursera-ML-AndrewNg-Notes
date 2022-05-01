%% 2 Linear regression with one variable
%% 2.1 plotting the data

% ex1data1.txt
data = load('./ex1data1.txt');
x = data(:, 1); % �˿�
y = data(:, 2); % ����
m = length(y);
figure;
plot(x, y, 'rx', 'markerSize', 10);
ylabel('profit in $10,000s');
xlabel('population of city in 10,000s');
%% 2.2 gradient decent
%% 2.2.1 update equations
% $$j(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2\\h_\theta(x) 
% = \theta^Tx = \theta_01 + \theta_1x_1\\\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) 
% - y^{(i)})x^{(i)}_j$$
% 
% ע�� $x^{\left(j\right)}$ �Ƕ�Ӧ $\theta_j$ ���������ݣ������Ѿ�ͳһ������˼ά��
%% 2.2.2 implementation
% ����һ��ά�ȸ� x�����ڶ�Ӧ $\theta_0$

X = x; % ����
x = [ones(m, 1), data(:, 1)];
x
theta = zeros(2, 1);
theta
iterations = 1500;
alpha = 0.01;
%% 2.2.3 computing the cost $j\left(\theta \right)$

compute_cost(x, y, theta)
%% 2.2.4 gradient descent
% we minimize the value of _J_(_��_) by changing the values of the vector _��_, 
% not by changing _X _or _y_  

x
y
theta = [0.11; 0.14]
theta = zeros(2, 1)
h     = x * theta
cost  = sum(x .* (h - y), 1);
cost.'
% ������ϵ���״��л�Զ��庯�� gd
trained_theta = gd(x, y, 1000, 0.01)
x * trained_theta
%% 
% *ע������ X �� x ��ʹ�ã�*X ��һά��x ������һ��ά��

figure;
plot(X, y, 'rx', X, x * trained_theta, 'b', 'MarkerSize', 10);
%% 2.4 *Visualizing *$J\left(\theta \right)$
% $J\left(\theta \right)$����ʧ���������ǿ��ӻ���ʧ����������Ҫ׼����������ʧ����

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = compute_cost(x, y, t);
    end
end

J_vals = J_vals'; % �˴�ע��
%% 
% ����ͼ

figure;
surf(theta0_vals, theta1_vals, J_vals); % ֵΪ��ʧ��
xlabel('\theta_0'); ylabel('\theta_1'); % ��֧�� latex
%% 
% �ȸ���ͼ

figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20));
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(trained_theta(1), trained_theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
%% 3. *Linear regression with multiple variables*
% $$h_\theta = X\theta = x_1\theta_1 + x_2\theta_2 + ..... x_n\theta_n$$
% 
% n ��������
%% 3.1 *Feature Normalization*
%%
data = load('./ex1data2.txt');
%% 
% the size of the house,  the number of bedrooms, the price of the house

data
%% 
% ���ַ������Զ���ڷ���������������������( �����������������ʦ�õȸ���ͼ�Ľ���

m = mean(data, 1) % ���ŵ�һά��(��)�������ۼӣ����Ե�һά����ʧ��
%% 
% ��������(��ʵ���Ҫ�� mean �� std ����)��ע�⣬y ����Ҳ������

% m = mean(data, 1);
% s = std(data, 0, 1);
% X = (data - mean(data, 1)) ./ std(data, 0, 1)
[X, mu_, sigma] = feature_normalize(data)
%% 3.2 *Gradient Descent*
% ����һ�£��� compute_cost �����ͬ

J = computeCostMulti(x, y, theta)
%% 
% �õ��������Իع�������Ȳ���һ�£�û������

[theta_multi, J_history] = gradientDescentMulti(x, y, theta, 0.01, 1000)
%% 
% ���ﲻ�����Ӷ����ȫΪ 1 ����

x_multi = [X(:, 1), X(:, 2)]
y_multi = X(:, 3)
theta = zeros(size(x_multi, 2), 1);
[theta_multi, J] = gradientDescentMulti(x_multi, y_multi, theta, 0.01, 1500);
figure;
plot(1:1500, J(1:1500), 'r');
xlabel('number of iters'); ylabel('cost J');
%% 3.2.1 *Optional (ungraded) exercise: Selecting learning rates*
% We recommend trying values of the learning rate _�� _on a log-scale, at multiplicative 
% steps of about 3 times the previous value
% 
% ժ��ʵ�� ex1 ԭ��

theta = zeros(2, 1);
alpha = 0.01;
num_iters = 1500;
[theta_multi, J1] = gradientDescentMulti(x, y, theta, alpha, num_iters);
plot(1:50, J1(1:50), 'b');
xlabel('number of iterations');
ylabel('cost J');
hold on;
alpha = alpha / 3;
[theta_multi, J2] = gradientDescentMulti(x, y, theta, alpha, num_iters);
plot(1:50, J2(1:50), 'r');
alpha = alpha / 3 / 3;
[theta_multi, J3] = gradientDescentMulti(x, y, theta, alpha, num_iters);
plot(1:50, J3(1:50), 'g');
hold off;
%% 
% ���Կ�����ͬѧϰ�ʵ�Ч����
% 
% ���Ǵ˴���û�г��Դ��ѧϰ�ʵ�Ӱ�죬����Ԥ�� cost �������ǵĽ��
%% 3.3 *Normal Equations*

add_ones_x_multi = [ones(length(y_multi), 1), x_multi];
perfect_theta = normalEqn(x_multi, y_multi)
%% 
% ���ݶ��½��õ��Ĳ�������Ԥ��

theta = zeros(size(x_multi, 2), 1);
[theta_multi, J] = gradientDescentMulti(x_multi, y_multi, theta, 0.01, 1500);
[1650, 3] * theta_multi
%% 
% �����淽�̵õ��Ĳ�������Ԥ��

[1650, 3] * perfect_theta
%% 
% 
%% �����Զ�����
%%
function cost = compute_cost(x, y, theta)
    % x .* y ��ӦԪ�����
    h = x * theta;
    m = length(x);
    c = (h - y) .^ 2;
    cost = sum(c) / (2 * m);
end
%% 
% $$j(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2\\h_\theta(x) 
% = \theta^Tx = \theta_01 + \theta_1x_1\\\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) 
% - y^{(i)})x^{(i)}_j$$
% 
% ��ע��һ�¾�����״��

% �ݶ��½�����ϳ����� 
function theta = gd(x, y, iterations, alpha)
    theta = zeros(2, 1);
    m = length(y);
    
    for i = 1 : iterations
        h    = x * theta;
        cost = sum((h - y) .* x, 1); % �����мӺͣ����ǵ�һά��
        theta = theta - alpha * cost.' ./ m; % ��ת�ûᷢ���㲥
    end
end


%% 
% 
% 
% ��������

function [X_norm, mu_, sigma] = feature_normalize(x)
    % ���ﻹô��
    X_norm = x;
    % ��ȡƽ������׼��
    mu_     = zeros(1, size(x, 2));
    sigma  = zeros(1, size(x, 2));
    
    % mu sigma �����������״
    % ��� A Ϊ������ô mean(A) ���ذ���ÿ�о�ֵ����������
    mu_    = mean(x);
    sigma  = std(x);
    
    X_norm = (X_norm - mu_) ./ sigma;
end
%% 
% 
% 
% 
% 
% J_history ����ʧ�����飬��¼��ʧ�仯
% 
% PS: 
% 
% # sum �����þ����ڻ�����
% # dev �������ע�� shape

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
    m = length(y);
    J_history = zeros(num_iters, 1); % zero(x) Ĭ���� x * x ����
    for i = 1 : num_iters
        
        theta = theta - alpha / m * X' * (X * theta - y); % ������ã��õ��Ǿ����ڻ�
        
        % dev = sum((X * theta - y) .* X, 1); % ע������ shape(1, size(X, 2)) Ҳ����������
        % theta = theta - alpha .* dev' ./ m; % theta shape(size(X, 2), 1) ����Ҫת��
        
        J_history(i, 1) = computeCostMulti(X, y, theta);
    end
end
%% 
% 
% 
% ��ʵ�� compute_cost Ҳ���ԣ�����ֱ�ӿ����������ˡ�
% 
% �� cost ����д�� $\frac{1}{2m}{\left(X\theta -y\right)}^T \left(X\theta -y\right)$����ʽ����ͬ��
% 
% $$\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2\\$$

function J = computeCostMulti(X, y, theta)
    m = length(X);
    J = (X * theta - y)' * (X * theta - y) / (2 * m);
end

%% 
% 
% 
% ������������
% 
% we still need to add a column of 1��s to the X matrix

function theta = normalEqn(X, y)
    theta = pinv(X' * X) * X' * y;
end