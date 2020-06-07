%% Machine Learning 4th Lab Assignment - Naive Bayes Classifier
% Francisco Melo - 84053
%
% Rodrigo Rego - 89213
%
% Group Number - 1
%
% Shift - Sexta 14h
%
% 13/11/2018
%% 2 - A simple example
%% 2.1

clear all; close all; clc;
load data1.mat

figure()
for i = 1:length(ytrain)
    if ytrain(1,i) == 1 || ytest(1,i) == 1
        scatter(xtrain(1,i), xtrain(2,i), 'bo'); hold on;
        scatter(xtest(1,i), xtest(2,i), 'bo'); hold on;
    elseif ytrain(1,i) == 2 || ytest(1,i) == 2
        scatter(xtrain(1,i), xtrain(2,i), 'rx'); hold on;
        scatter(xtest(1,i), xtest(2,i), 'rx'); hold on;
    elseif ytrain(1,i) == 3 || ytest(1,i) == 3
        scatter(xtrain(1,i), xtrain(2,i), 'g*'); hold on;
        scatter(xtest(1,i), xtest(2,i), 'g*'); hold on;
    end
end
title('\textbf{Scatter Plot of the Training and Test Data}', 'Interpreter', 'latex');
xlabel('$x_{1}$', 'Interpreter', 'latex');
ylabel('$x_{2}$', 'Interpreter', 'latex');
l1 = plot([NaN,NaN], 'bo'); % Dummy Plot for Legend
l2 = plot([NaN,NaN], 'rx'); % Dummy Plot for Legend
l3 = plot([NaN,NaN], 'g*'); % Dummy Plot for Legend
legend([l1, l2, l3], {'Class 1', 'Class 2','Class 3'},'location','northwest');
axis([-5 5 -3 7]);
grid on;