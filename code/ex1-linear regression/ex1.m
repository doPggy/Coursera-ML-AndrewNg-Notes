%% 
%% 2 Linear regression with one variable
%% 2.1 plotting the data
%%
% ex1data1.txt
data = load('./ex1data1.txt');
x = data(:, 1); % �˿�
y = data(:, 2); % ����
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

cost_func(x, y, theta, m)
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
train_theta = gd(x, y, 1000, 0.01)
x * train_theta
%% 
% *ע������ X �� x ��ʹ�ã�*X ��һά��x ������һ��ά��

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
% ���ַ������Զ���ڷ���������������������( �����������������ʦ�õȸ���ͼ�Ľ���

m = mean(data, 1); % ���ŵ�һά��(��)�������ۼӣ����Ե�һά����ʧ��

%% 
% 
% 
% 
% 
% 
%% �����Զ�����
%%
function cost = cost_func(x, y, theta, m)
    % x .* y ��ӦԪ�����
    h = x * theta;
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
%