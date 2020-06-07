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
clear all; close all; clc;
load data1.mat

% Auxiliary counters
ct_c1 = 1;
ct_c2 = 1;
ct_c3 = 1;

% Groups Training Data by Class Array
for i = 1:length(ytrain)
    if ytrain(1,i) == 1
        class_1(:, ct_c1) = xtrain(:,i);
        ct_c1 = ct_c1 + 1;
    elseif ytrain(1,i) == 2
        class_2(:, ct_c2) = xtrain(:,i);
        ct_c2 = ct_c2 + 1;
        
    elseif ytrain(1,i) == 3
        class_3(:, ct_c3) = xtrain(:,i);
        ct_c3 = ct_c3 + 1;
    end
end

% Computation of mean vector of training data for each class and feature 
mean_class1 = mean(class_1')';
mean_class2 = mean(class_2')';
mean_class3 = mean(class_3')';

N = length(class_1(1,:)); % Number of data points

% Computation of variance of the training data for each class and feature.
% Using var(...,1) allows for computing the the maximum-likelihood
% estimator of the Normal Distribution (biased sample variance), instead of
% the corrected variance (unbiased sample variance).
var_class1 = [var(class_1(1,:),1); var(class_1(2,:),1)];
var_class2 = [var(class_2(1,:),1); var(class_2(2,:),1)];
var_class3 = [var(class_3(1,:),1); var(class_3(2,:),1)];

x1 = xtest(1,:);
x2 = xtest(2,:);

% Computation of the normal distribution probabilities for each class, for
% the Testing Data [x1; x2] using Naive Bayes assumption, where we multiply
% the normal probabilities of both features, considered independent
px_C1 = normpdf(x1, mean_class1(1,1), sqrt(var_class1(1,1))) .* ...
        normpdf(x2, mean_class1(2,1), sqrt(var_class1(2,1)));

px_C2 = normpdf(x1, mean_class2(1,1), sqrt(var_class2(1,1))) .* ...
        normpdf(x2, mean_class2(2,1), sqrt(var_class2(2,1)));

px_C3 = normpdf(x1, mean_class3(1,1), sqrt(var_class3(1,1))) .* ...
        normpdf(x2, mean_class3(2,1), sqrt(var_class3(2,1)));

[l, c] = size(px_C1);

% Computation of the Classification of the Test Data
for i = 1:c
    
    % Groups in a column the probability of each point for each class,
    % where the row index coincides with the class number
    aux_v = [px_C1(1,i); px_C2(1,i); px_C3(1,i)]; 
    
    % Computes the maximum probability (M) between each class, and returns
    % the class number (I)
    [M, I] = max(aux_v); 
    
    % Creates the Classification vector (output of the naive bayes method)
    naive_bayes(i) = I;
    
end

% Computes a binary vector, with 0s or 1s, for each index, where 0 means
% there was not an error in the classification, and 1 means there was an
% error in the classification of the point associated to that index
binary_error = naive_bayes ~= ytest;

% Computes the error percentage based on the binary vector
error_percentage = 100*(sum(binary_error)/length(binary_error));

% Creates the vector of test pattern index
xt = 1:1:length(naive_bayes);

% Plots the Classifications of the Test Data as a function of the test
% pattern index
figure()
scatter(xt, naive_bayes, 'rx');
title('\textbf{Classifications of the Test Data}', 'Interpreter', 'latex');
xlabel('Index', 'Interpreter', 'latex');
ylabel('Class', 'Interpreter', 'latex');
axis([1 151 1 3]);
grid on;

% Prints the error percentage
fprintf('Error Percentage = %g%%\n', error_percentage);