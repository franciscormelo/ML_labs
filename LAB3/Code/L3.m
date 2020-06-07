%% Machine Learning 3rd Lab Assignment - Neural Networks
% Francisco Melo - 84053
%
% Rodrigo Rego - 89213
%
% Group Number - 1
%
% Shift - Sexta 14h
%
% 02/11/2018
%% Classification

%% Testing commands
close all; clear all; clc;

load digits;
size(X)
size(T)

show_digit(X, 500);

%% Gradient Method with fixed Step Size and Momentum
close all; clear all; clc;
load digits;

net = patternnet([15]);             % number of units in hidden layer
net.performFcn='mse';               % mean squarred normalized error performance function
net.layers{1}.transferFcn='tansig'; % activation function: hyperbolic tangent (layer 1 - hidden)
net.layers{2}.transferFcn='tansig'; % activation function: hyperbolic tangent (layer 2 - output)

net.divideFcn='divideind'; 
net.divideParam.trainInd=1:400;   % 400 patterns for training
net.divideParam.testInd=401:560;  % 160 patterns for testing

% Gradient Method with fixed Step Size and Momentum
net.trainFcn = 'traingdm'
net.trainParam.lr=4.25;         % learning rate
net.trainParam.mc=0.5;          % Momentum constant
net.trainParam.show=10000;      % # of epochs in display
net.trainParam.epochs=10000;    % max epochs
net.trainParam.goal=0.05;       % training goal
[net,tr] = train(net,X,T);

%% Gradient Method with adaptive Step Size and Momentum
close all; clear all; clc;
load digits;

net = patternnet([15]);             % number of units in hidden layer
net.performFcn='mse';               % mean squarred normalized error performance function
net.layers{1}.transferFcn='tansig'; % activation function: hyperbolic tangent (layer 1 - hidden)
net.layers{2}.transferFcn='tansig'; % activation function: hyperbolic tangent (layer 2 - output)

net.divideFcn='divideind';
net.divideParam.trainInd=1:400;   % 400 patterns for training
net.divideParam.testInd=401:560;  % 160 patterns for testing

net.trainFcn = 'traingdx'
net.trainParam.lr=1.25;          % learning rate
net.trainParam.mc=0.8;          % Momentum constant
net.trainParam.show=10000;      % # of epochs in display
net.trainParam.epochs=10000;    % max epochs
net.trainParam.goal=0.05;       % training goal
[net,tr] = train(net,X,T);

%% Confusion Matrix and Accuracy
close all; clear all; clc;
load digits;

net = patternnet([15]);             % number of units in hidden layer
net.performFcn='mse';               % mean squarred normalized error performance function
net.layers{1}.transferFcn='tansig'; % activation function: hyperbolic tangent (layer 1 - hidden)
net.layers{2}.transferFcn='tansig'; % activation function: hyperbolic tangent (layer 2 - output)

net.divideFcn='divideind';
net.divideParam.trainInd=1:400;   % 400 patterns for training
net.divideParam.testInd=401:560;  % 160 patterns for testing

net.trainFcn = 'traingdx'
net.trainParam.lr=1.25;          % learning rate
net.trainParam.mc=0.8;          % Momentum constant
net.trainParam.show=10000;      % # of epochs in display
net.trainParam.epochs=10000;    % max epochs
net.trainParam.goal=0.05;       % training goal
[net,tr] = train(net,X,T);

% Confusion and Accuracy for Testing Set
figure()
x_test=X(:,tr.testInd); 
t_test=T(:,tr.testInd); 
y_test = net(x_test); 
plotconfusion(t_test,y_test);

% Confusion and Accuracy for Training Set
figure()
x_train=X(:,tr.trainInd); 
t_train=T(:,tr.trainInd); 
y_train = net(x_train); 
plotconfusion(t_train,y_train);

% MSE of Training and Testing Errors
fprintf('Training Error (mse): %g\n', tr.best_perf);
fprintf('Testing Error (mse): %g\n', tr. best_tperf);

%% Regression

%% Training without a validation set
clear all; close all; clc;
load regression_data.mat

net = fitnet(20); 
net.performFcn='mse';
net.layers{2}.transferFcn='purelin'; 

net.divideFcn='divideind';
net.divideParam.trainInd=1:85;   
net.divideParam.testInd=86:100;

net.trainFcn = 'trainlm';
net.trainParam.show=3000;      
net.trainParam.epochs=3000;    
net.trainParam.goal=0.005; 
[net,tr] = train(net,X,T);

x = -1:0.01:1;

figure()
f1 = net(x);
plot(x, f1, 'k'); hold on;
plot(X(1:85), T(1:85), 'rx'); hold on;
plot(X(86:100), T(86:100), 'b*');
title('Neural Network Regression - Without Validation Set');
xlabel('x');
ylabel('f(x)');
legend('Estimated Function', 'Training Data', 'Testing Data', 'Location', 'best');
grid on;

save('output_without_val', 'f1');

% set(0,'defaultfigurecolor',[255 255 255]);


%% Training with a validation set
clear all; close all; clc;
load regression_data.mat
load output_without_val

net = fitnet(20); 
net.performFcn='mse';
net.layers{2}.transferFcn='purelin'; 

net.divideFcn='divideind';
net.divideParam.trainInd=1:70;
net.divideParam.valInd=71:85;  
net.divideParam.testInd=86:100;

net.trainFcn = 'trainlm';
net.trainParam.show=3000;      
net.trainParam.epochs=3000;    
net.trainParam.goal=0.005; 
[net,tr] = train(net,X,T);

x = -1:0.01:1;

figure()
plot(x, net(x), 'k'); hold on;
plot(x, f1, 'm'); hold on;
plot(X(1:70), T(1:70), 'rx'); hold on;
plot(X(71:85),T(71:85), 'g*'); hold on;
plot(X(86:100), T(86:100), 'b*');
title('Neural Network Regression - With Validation Set');
xlabel('x');
ylabel('f(x)');
legend('Estimated Function', 'Without Validation', 'Training Data', 'Validation Data', 'Testing Data', 'Location', 'nw');
grid on;
% set(0,'defaultfigurecolor',[255 255 255]);

%% Training without a validation set (50 UNITS)
clear all; close all; clc;
load regression_data.mat

net = fitnet(50); 
net.performFcn='mse';
net.layers{2}.transferFcn='purelin'; 

net.divideFcn='divideind';
net.divideParam.trainInd=1:85;   
net.divideParam.testInd=86:100;

net.trainFcn = 'trainlm';
net.trainParam.show=3000;      
net.trainParam.epochs=3000;    
net.trainParam.goal=0.005; 
[net,tr] = train(net,X,T);

x = -1:0.01:1;

figure()
plot(x, net(x), 'k'); hold on;
plot(X(1:85), T(1:85), 'rx'); hold on;
plot(X(86:100), T(86:100), 'b*');
title('Neural Network Regression - Without Validation Set');
xlabel('x');
ylabel('f(x)');
legend('Estimated Function', 'Training Data', 'Testing Data', 'Location', 'best');
grid on;
% set(0,'defaultfigurecolor',[255 255 255]);

%% Training with a validation set (50 UNITS)
clear all; close all; clc;
load regression_data.mat

net = fitnet(50); 
net.performFcn='mse';
net.layers{2}.transferFcn='purelin'; 

net.divideFcn='divideind';
net.divideParam.trainInd=1:70;
net.divideParam.valInd=71:85;  
net.divideParam.testInd=86:100;

net.trainFcn = 'trainlm';
net.trainParam.show=3000;      
net.trainParam.epochs=3000;    
net.trainParam.goal=0.005; 
[trained_net,tr] = train(net,X,T);

x = -1:0.01:1;

figure()
plot(x, trainer_net(x), 'k'); hold on;
plot(X(1:70), T(1:70), 'rx'); hold on;
plot(X(71:85),T(71:85), 'g*'); hold on;
plot(X(86:100), T(86:100), 'b*');
title('Neural Network Regression - With Validation Set');
xlabel('x');
ylabel('f(x)');
legend('Estimated Function', 'Training Data', 'Validation Data', 'Testing Data', 'Location', 'nw');
grid on;
% set(0,'defaultfigurecolor',[255 255 255]);