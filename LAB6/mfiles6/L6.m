%% Machine Learning 6th Lab Assignment - Optimization and Generalization
% Francisco Melo - 84053
%
% Rodrigo Rego - 89213
%
% Group Number - 1
%
% Shift - Sexta 14h
%
% 14/12/2018

%% Dataset 2
%% Neural Network

close all; clear all; clc;
load dataset2.mat;

X=[Xtrain' Xtest'];
T=[Ytrain' Ytest'];

net = patternnet([15]);             % number of units in hidden layer
net.performFcn='mse';               % mean squarred normalized error performance function
net.layers{1}.transferFcn='tansig'; % activation function: hyperbolic tangent (layer 1 - hidden)
net.layers{2}.transferFcn='softmax'; % activation function: softmax(layer 2 - output)

net.divideFcn='divideind';
net.divideParam.trainInd=1:650;
net.divideParam.valInd=651:921; %validation set
net.divideParam.testInd=922:1151;  

net.trainFcn = 'traingdx'
net.trainParam.lr=0.5;          % learning rate
net.trainParam.mc=0.6;          % Momentum constant
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
fprintf('Validation (mse): %g\n', tr.best_vperf);
fprintf('Testing Error (mse): %g\n',tr.best_tperf);


% Performance
labelTest = y_test';
labelTrain = y_train';

[Xp,Yp,Tp,AUC,Opto] = perfcurve(Ytest,labelTest,1);

figure()
plot(Xp,Yp, 'linewidth', 1.5); hold on;
plot(Opto(1), Opto(2), 'ro');
xlabel('False positive rate', 'interpreter', 'latex');
ylabel('True positive rate', 'interpreter', 'latex');
title('ROC for Classification by Neural Network - Dataset 2', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;


