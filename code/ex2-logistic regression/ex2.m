%% 
%% Logistic Regression
% build a logistic regression model to predict whether a student gets admitted 
% into a university  
%% 1.1 *Visualizing the data*
% The first two columns contains the exam scores and the third column contains 
% the label.

data = load('./ex2data1.txt')
x = data(:, [1, 2]); y = data(:, 3);
x, y
%% 
% �ҵ��������

pos = find(y == 1)
neg = find(y == 0)
%% 
% Ҫ������������������ x �� y��
% 
% ͬʱҪ�������ฺ��

figure;
% x �� exam 1 �ķ�����y �� exam 2 �ķ���
plot(x(pos, 1), x(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
xlabel('exam 1 score'); ylabel('exam 2 score');
hold on;
plot(x(neg, 1), x(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7); legend('admitted', 'not admitted'); hold off; 
%% 1.2 implementation
%% 1.2.1 *Warmup exercise: sigmoid function*
% $$h_\theta(x) = g(\theta^Tx)\\g(z) = \frac{1}{1 + e^{-z}}$$

sigmoid(0)
%% 1.2.2 *Cost function and gradient*
% cost ���£�
% 
% $$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log 
% \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log 
% \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$$
% 
% ����ǻ����ģ�ԭ����˼�����һ���ֶ��жϡ�ͬʱ gradient ���£�
% 
% $$\theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}{{\left( 
% {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i 
% \right)} \right)}}\mathop{x}_{j}^{(i)}$$
% 
% �����Իع���ݶ��½���ʽ����һ������ h ��һ����

initial_theta = zeros(size(x, 2) + 1, 1); % �� \theta_0
X = [ones(length(y), 1), x] % \theta_0 ��
costFunction(initial_theta, X, y)
%% 1.2.3 *Learning parameters using _*fminunc_

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
cost,theta
%% 
% �ø����ʵĳ�ʼ�������ܹ��õ���С����ʧ

test_theta = [-24; 0.2; 0.2];
[cost_test, grad] = costFunction(test_theta, X, y);
cost_test
X(:, 2:3)
plot_x = [min(X(:,2))-2,  max(X(:,2))+2]
plot_y = (-1 ./ theta(3)).* (theta(2) .* plot_x + theta(1))
%% 
% ����Ҫ�󻭾��߽߱磬�һ��ǿ������������ˣ���������
% 
% $$\theta_0 + \theta_1x_1 + \theta_1x_2 = 0$$
% 
% �����ȥ���Σ�������⡣

plot_x = [min(X(:,2))-2,  max(X(:,2))+2]; 
plot_y = [];

%% 1.2.4 *Evaluating logistic regression*

sigmoid([1 45, 85] * theta)
p = predict(theta, X)
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
%% *2 Regularized logistic regression*
%% *2.1 Visualizing the data*
%%
data = load('ex2data2.txt')
x = data(:, 1:2)
y = data(:, 3)
m = length(y);
X = [ones(m, 1), x]
pos = (y == 1)
neg = (y == 0)
figure;
% x �� exam 1 �ķ�����y �� exam 2 �ķ���
plot(x(pos, 1), x(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
xlabel('microchip test 1'); ylabel('microchip test 2');
hold on;
plot(x(neg, 1), x(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7); legend('y = 1', 'y = 0'); hold off; 
%% 2.2 *Feature mapping*
% ���ô˺������Խ��������ԵĴ��۱߽硣
% 
% $$mapFeature(x) = [1, x_1, x_2, x_1^2, x_1x_2,x_2^2,x_1^3,..., x_1x_2^5, 
% x_1^6]^T$$
% 
% ����� x1, x2 ������ feature �Ķ�Ӧ����ȡֵ���ڱ���ʵ���о��� microchip test 1 �ķ������� Micochip 
% test 2 �ķ�����
% 
% ���Ļ���ʹ��
%% 2.3 *Cost function and gradient*
% ����������߼��ع���ۺ������£�
% 
% $$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log 
% \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log 
% \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta 
% _{j}^{2}}$$
% 
% ע�⣬���ͷ� $\theta_0$�����Թ��������ݶ��½�Ҫ�ֿ�������
% 
% $${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})\\{\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( 
% i \right)}}+\frac{\lambda }{m}{\theta_j}]$$
% 
% 

% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,2), X(:,3))
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
cost, grad
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');
%% 2.3.1 *Learning parameters using *fminunc

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
theta, J, exit_flag
%% 2.4 *Plotting the decision boundary*

plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
%% 
% Compute accuracy on our training set

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
%% 2.5 *Optional  exercises*
% �ı� $\lambda$ ���۲���߽߱�
% 
% ���� $\lambda$ ���࣬����ϡ�

lambda = 0;
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
%% 
% ���� $\lambda$ ���� ��Ƿ���

lambda = 100;
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
%% ���ղ��õĵط�
% # plotDecisionBoundary �У�����ֱ�߽߱�ͷ����Ա߽�
% # predict �ı�д�У���ΰѸ���תΪ 0 �� 1 �ķ��ࡣ
%% �����Զ���
% $$g(z) = \frac{1}{1 + e^{-z}}$$
%%
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
%% 
% $$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log 
% \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log 
% \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}\\\frac{1}{m}\sum\limits_{i=1}^{m}{{\left( 
% {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i 
% \right)} \right)}}\mathop{x}_{j}^{(i)}$$

function [J, grad] = costFunction_my(theta, X, y) % ��� ok
    m = length(y);
    h_theta = sigmoid(X * theta);
    grad = sum((h_theta - y) .* X, 1) ;
    J = sum(-y .* log(h_theta) - (1 - y) .* log(1 - h_theta), 1) ./ m;
end

function [J, grad] = costFunction(theta, X, y)
    m = length(y);
    J = 0;
    grad = zeros(size(theta));
    
    h_theta = sigmoid(X * theta);
    J       = sum(-y .* log(h_theta) - (1 - y) .* log(1 - h_theta), 1) ./ m;
    
    grad = (X' * (h_theta - y)) ./ m; % ������Ҫ��˼ά
    %grad = sum((h_theta - y) .* X, 1) ;
end

%% 
% 

function p = predict(theta, X)
    m = size(X, 1); % Number of training examples
    
    % You need to return the following variables correctly
    p = zeros(m, 1);
    
    % ����Ļ����� if��Ҫ�����·�ʽ�� >= 0.5 ��������
    k = sigmoid(X * theta) >= 0.5;
    p(k) = 1;
end

%% 
% 

function out = mapFeature(X1, X2)
    % MAPFEATURE Feature mapping function to polynomial features
    %
    %   MAPFEATURE(X1, X2) maps the two input features
    %   to quadratic features used in the regularization exercise.
    %
    %   Returns a new feature array with more features, comprising of 
    %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    %
    %   Inputs X1, X2 must be the same size
    %
    
    degree = 6;
    out = ones(size(X1(:,1)));
    for i = 1:degree
        for j = 0:i
            out(:, end+1) = (X1.^(i-j)).*(X2.^j);
        end
    end

end

%% 
% 
% 
% $$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log 
% \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log 
% \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta 
% _{j}^{2}}\\{\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})\\{\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( 
% i \right)}}+\frac{\lambda }{m}{\theta_j}]$$

function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y);
    grad = zeros(size(theta));
    J = costFunction(theta, X, y) + theta(2:end, :)' * theta(2:end, :) * lambda / (2 * m);
    h_theta = sigmoid(X * theta);
    reg_theta = theta; % ��ֵ�����
    reg_theta(1) = 0;
%     theta_1=[0;theta(2:end)];    % �Ȱ�theta(1)�õ�������������
    grad = X' * (h_theta - y) / m + lambda / m * reg_theta; % ̫���˸������һ��
end
%% 
% 

function plotDecisionBoundary(theta, X, y)
    %PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    %the decision boundary defined by theta
    %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    %   positive examples and o for the negative examples. X is assumed to be 
    %   a either 
    %   1) Mx3 matrix, where the first column is an all-ones column for the 
    %      intercept.
    %   2) MxN, N>3 matrix, where the first column is all-ones
    
    % Plot Data
    plotData(X(:,2:3), y);
    hold on
    
    if size(X, 2) <= 3
        % Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
    
        % Calculate the decision boundary line
        plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    
        % Plot, and adjust axes for better viewing
        plot(plot_x, plot_y)
        
        % Legend, specific for the exercise
        legend('Admitted', 'Not admitted', 'Decision Boundary')
        axis([30, 100, 30, 100])
    else
        % Here is the grid range
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);
    
        z = zeros(length(u), length(v));
        % Evaluate z = theta*x over the grid
        for i = 1:length(u)
            for j = 1:length(v)
                z(i,j) = mapFeature(u(i), v(j))*theta;
            end
        end
        z = z'; % important to transpose z before calling contour
    
        % Plot z = 0
        % Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2)
    end
    hold off

end

function plotData(X, y)
    %PLOTDATA Plots the data points X and y into a new figure 
    %   PLOTDATA(x,y) plots the data points with + for the positive examples
    %   and o for the negative examples. X is assumed to be a Mx2 matrix.
    
    % Create New Figure
    figure; hold on;
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Plot the positive and negative examples on a
    %               2D plot, using the option 'k+' for the positive
    %               examples and 'ko' for the negative examples.
    %
    pos = y == 1; neg = y == 0;
    % Plot Examples
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

    
    % =========================================================================

    hold off;

end

%% 
%