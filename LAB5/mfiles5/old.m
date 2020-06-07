%% Machine Learning 5th Lab Assignment - Support Vector Machines
% Francisco Melo - 84053
%
% Rodrigo Rego - 89213
%
% Group Number - 1
%
% Shift - Sexta 14h
%
% 30/11/2018
%% 4 Experiments
%% 4.1 Polynomial kernel

clear all; close all; clc
load spiral.mat

% N=100;
%
% i=randperm(N);
%
% xTrain = X(i(1:(0.7*N)),:); % 70% of training data
% xTest =X(i((0.7*N):N),:);
%
% yTrain=Y(i(1:(0.7*N)));
%
% yTest=Y(i((0.7*N):N));

load good_data.mat;
p_test=1:1:15;


for i=1:length(p_test)
    SVMStruct = svmtrain(xTrain,yTrain,'kernel_function','polynomial', 'polyorder'...
        ,p_test(i),'method','QP','boxconstraint',10^4,'Showplot',true);
    
    Group = svmclassify(SVMStruct,xTest,'Showplot',true);
    
    error(i) = (sum((Group~=yTest))/length(yTest))*100;
    n_sv(i) = length(SVMStruct.SupportVectors);
    
    fprintf('P = %g Error = %g %%\n',p_test(i), error(i));
    fprintf('P = %g Number of support Vector = %g\n\n',p_test(i),n_sv(i));
    
    
end
figure();
subplot(2,1,1);
plot(p_test,error);
grid on;
title('Error in respect to p','interpreter','latex');
xlabel('p - Polynomial order','interpreter','latex');
ylabel('Error','interpreter','latex');
xlim([1 15]);
subplot(2,1,2);
plot(p_test,n_sv,'r');
title('Number of Support Vectors in respect to p','interpreter','latex');
xlabel('p - Polynomial order','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
xlim([1 15]);
grid on;


%% 4.2 -
clear all; close all; clc
load spiral.mat
load good_data.mat;

s_test=0.1:0.1:2;
s=0.9;

for i=1:length(s_test)
    SVMStruct = svmtrain(xTrain,yTrain,'kernel_function','rbf', 'rbf_sigma'...
        ,s_test(i),'method','QP','boxconstraint',10^4,'Showplot',true);
    
    Group = svmclassify(SVMStruct,xTest,'Showplot',true);
    
    error(i) = (sum((Group~=yTest))/length(yTest))*100;
    n_sv(i) = length(SVMStruct.SupportVectors);
    
    fprintf('sigma = %g Error = %g %%\n',s_test(i), error(i));
    fprintf('sigma = %g Number of support Vector = %g\n\n',s_test(i),n_sv(i));
    
   
end
figure();
subplot(2,1,1);
plot(s_test,error);
grid on;
title('Error in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Error','interpreter','latex');
xlim([0 2]);
subplot(2,1,2);
plot(s_test,n_sv,'r');
title('Number of Support Vectors in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
xlim([0 2]);
grid on;

%% 4.3
close all; clear all;clc
load chess33.mat

N=90;

i=randperm(N);

xTrain = X(i(1:(0.7*N)),:); % 70% of training data
xTest =X(i((0.7*N):N),:);

yTrain=Y(i(1:(0.7*N)));

yTest=Y(i((0.7*N):N));


%%
close all; clear all; clc;
load good_data2.mat
s_test=0.1:0.1:4;


for i=1:length(s_test)
    SVMStruct = svmtrain(xTrain,yTrain,'kernel_function','rbf', 'rbf_sigma'...
        ,s_test(i),'method','QP','boxconstraint',Inf,'Showplot',true);
    
    Group = svmclassify(SVMStruct,xTest,'Showplot',true);
    
    error(i) = (sum((Group~=yTest))/length(yTest))*100;
    n_sv(i) = length(SVMStruct.SupportVectors);
    
    fprintf('sigma = %g Error = %g %%\n',s_test(i), error(i));
    fprintf('sigma = %g Number of support Vector = %g\n\n',s_test(i),n_sv(i));
    
    
end
figure();
subplot(2,1,1);
plot(s_test,error);
grid on;
title('Error in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Error','interpreter','latex');
xlim([0.1 2]);
subplot(2,1,2);
plot(s_test,n_sv,'r');
title('Number of Support Vectors in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
xlim([0.1 2]);
grid on;

%% 4.4
close all; clear all; clc

load chess33n.mat;
N=90;

i=randperm(N);

xTrain = X(i(1:(0.7*N)),:); % 70% of training data
xTest =X(i((0.7*N):N),:);

yTrain=Y(i(1:(0.7*N)));

yTest=Y(i((0.7*N):N));

figure()
plot(X(:,1), X(:,2),'+')
%%
close all; clear all; clc;
load good_data3.mat

s=0.5;

SVMStruct = svmtrain(xTrain,yTrain,'kernel_function','rbf', 'rbf_sigma'...
    ,s,'method','QP','boxconstraint',Inf,'Showplot',true);

Group = svmclassify(SVMStruct,xTest,'Showplot',true);

error = (sum((Group~=yTest))/length(yTest))*100;
n_sv = length(SVMStruct.SupportVectors);

fprintf('Error = %g %%\n',error);
fprintf('Number of support Vector = %g\n\n',n_sv);

%% 4.5
close all; clear all; clc;

load good_data3.mat

boxc=[10^-4 10^-3 10^-2 10^-1.5 10^-1 1 10 10^1.5 10^2 10^2.5 10^3 10^3.5 10^4 ];

s=0.9;

for i=1:length(boxc)
    
    SVMStruct = svmtrain(xTrain,yTrain,'kernel_function','rbf', 'rbf_sigma'...
        ,s,'method','QP','boxconstraint',boxc(i),'Showplot',true);
    
    Group = svmclassify(SVMStruct,xTest,'Showplot',true);
    
    error(i) = (sum((Group~=yTest))/length(yTest))*100;
    n_sv(i) = length(SVMStruct.SupportVectors);
    
    fprintf('boxconstraint = %g Error = %g %%\n',boxc(i), error(i));
    fprintf('boxconstraint = %g Number of support Vector = %g\n\n',boxc(i),n_sv(i));
    
end

lbox=log(boxc);
figure();
subplot(2,1,1);
plot(lbox,error);
grid on;
title('Error in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Error','interpreter','latex');
%xlim([0 10^9]);
subplot(2,1,2);
plot(lbox,n_sv,'r');
title('Number of Support Vectors in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
%xlim([0 10^9]);
grid on;
