%% Support Vector Machine (RBF) - dataset2.mat
clear all; close all; clc
load dataset2.mat
load roc_auc2

% Creation of a Validation Set
Xt = Xtrain(1:691,:);
Yt = Ytrain(1:691,:);
Xv = Xtrain(692:end,:);
Yv = Ytrain(692:end,:);

% Choice of Classifier Parameters (with a Validation Set)
s_test=0.1:0.05:2.5;

for i=1:length(s_test)
    SVMStruct = fitcsvm(Xt, Yt,'BoxConstraint',10,'KernelFunction','RBF', 'KernelScale',...
        s_test(i), 'Standardize', true, 'Solver', 'L1QP');
    
    Group = predict(SVMStruct, Xv);
    label = predict(SVMStruct, Xt);
    
    error(i) = (sum((Group~=Yv))/length(Yv))*100;
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    
    n_sv(i) = length(SVMStruct.SupportVectors);
end

figure();
subplot(2,1,1);
plot(s_test,error); hold on;
plot(s_test, errort);
grid on;
title('Error in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Error','interpreter','latex');
xlim([0 2.5]);
legend('Validation', 'Train');
subplot(2,1,2);
plot(s_test,n_sv,'r');
title('Number of Support Vectors in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
xlim([0 2]);
grid on;

clear error;
clear errort;
clear n_sv;

bc=[10^-4 10^-3 10^-2 10^-1 10^0 10^0.25 10^0.5 10^1 10^2 10^3 10^4];

for i=1:length(bc)
    SVMStruct = fitcsvm(Xt, Yt,'BoxConstraint',bc(i),'KernelFunction','RBF', 'KernelScale',...
        2.1, 'Standardize', true, 'Solver', 'L1QP');
    
    Group = predict(SVMStruct, Xv);
    label = predict(SVMStruct, Xt);
    
    error(i) = (sum((Group~=Yv))/length(Yv))*100;
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    n_sv(i) = length(SVMStruct.SupportVectors);
end

lbox=log10(bc);
figure();
subplot(2,1,1);
plot(lbox,error); hold on;
plot(lbox, errort);
grid on;
title('Error in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Error','interpreter','latex');
legend('Validation', 'Train');
subplot(2,1,2);
plot(lbox,n_sv,'r');
title('Number of Support Vectors in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
grid on;

% We chose sigma = 2.1 and BoxConstraint = 100

% Train SVM
tic
mdl = fitcsvm(Xtrain, Ytrain,'BoxConstraint',100,'KernelFunction','RBF', 'KernelScale',...
        2.1, 'Standardize', true, 'Solver', 'L1QP');
toc

labelv = predict(mdl, Xv);
valerror = (sum((labelv~=Yv))/length(Yv))*100;

% Performance
labelTest = predict(mdl, Xtest);
labelTrain = predict(mdl, Xtrain);
C1 = confusionmat(Ytest, labelTest);
C2 = confusionmat(Ytrain, labelTrain);
figure()
plotconfusion(Ytest', labelTest');
figure()
plotconfusion(Ytrain', labelTrain');


TP = C1(2,2); % True Positives
TN = C1(1,1); % True Negatives
FP = C1(1,2); % False Positives
FN = C1(2,1); % False Negatives

TP2 = C2(2,2); % True Positives
TN2 = C2(1,1); % True Negatives
FP2 = C2(1,2); % False Positives
FN2 = C2(2,1); % False Negatives


Accuracy = (TP+TN)/length(Ytest)*100;
Error = 100-Accuracy;
Sensitivity = (TP/(TP+FN))*100;
Specificity = (TN/(TN+FP))*100;
Precision = (TP/(TP+FP));
Recall = (TP/(TP+FN));
F_measure = 2*((Precision*Recall)/(Precision+Recall));

accuracyTrain = (TP2+TN2)/length(Ytrain)*100;
errorTrain = 100-accuracyTrain;

fprintf('Accuracy: %g%%\n', Accuracy);
fprintf('Error: %g%%\n', Error);
fprintf('Sensitivity: %g%%\n', Sensitivity);
fprintf('Specificity: %g%%\n', Specificity);
fprintf('Precision: %g\n', Precision);
fprintf('Recall: %g\n', Recall);
fprintf('F-measure: %g\n', F_measure);
fprintf('\nAccuracy Train: %g%%\n', accuracyTrain);
fprintf('Error Train: %g%%\n', errorTrain);

[Xp,Yp,Tp,AUC,Opto] = perfcurve(Ytest,labelTest,1);

figure()
plot(Xp,Yp, 'linewidth', 1.5); hold on;
plot(Opto(1), Opto(2), 'ro');
xlabel('False positive rate', 'interpreter', 'latex');
ylabel('True positive rate', 'interpreter', 'latex');
title('ROC for Classification by SVM (RBF) - Dataset 2', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;

ROC(1).data = [Xp Yp];
ROC(1).opto = Opto;
ROC(1).auc = AUC;

m = matfile('roc_auc2.mat','Writable',true);
save('roc_auc2', 'ROC');

%% Support Vector Machine (Polynomial) - dataset2.mat
clear all; close all; clc
load dataset2.mat
load roc_auc2

Xt = Xtrain(1:691,:);
Yt = Ytrain(1:691,:);
Xv = Xtrain(692:end,:);
Yv = Ytrain(692:end,:);

% Choice of Classifier Parameters (with validation set)
s_test=1:1:10;

for i=1:length(s_test)
    SVMStruct = fitcsvm(Xt, Yt,'BoxConstraint',10^4,'KernelFunction','polynomial', 'PolynomialOrder',...
        s_test(i), 'Standardize', true, 'Solver', 'L1QP');
    
    Group = predict(SVMStruct, Xv);
    label = predict(SVMStruct, Xt);
    
    error(i) = (sum((Group~=Yv))/length(Yv))*100;
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    
    n_sv(i) = length(SVMStruct.SupportVectors);
end

figure();
subplot(2,1,1);
plot(s_test,error); hold on;
plot(s_test, errort);
grid on;
title('Error in respect to $p$ - Polynomial Order','interpreter','latex');
xlabel('$p$','interpreter','latex');
ylabel('Error','interpreter','latex');
xlim([1 10]);
legend('Validation', 'Train');
subplot(2,1,2);
plot(s_test,n_sv,'r');
title('Number of Support Vectors in respect to $p$','interpreter','latex');
xlabel('$p$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
xlim([1 10]);
grid on;

clear error;
clear errort;
clear n_sv;

bc=[10^-4 10^-3 10^-2 10^-1 10^0 10^0.25 10^0.5 10^1 10^2 10^3 10^4 10^5 10^6];

for i=1:length(bc)
    SVMStruct = fitcsvm(Xtrain, Ytrain,'BoxConstraint',bc(i),'KernelFunction','polynomial', 'PolynomialOrder',...
        2, 'Standardize', true, 'Solver', 'L1QP');
    
    Group = predict(SVMStruct, Xv);
    label = predict(SVMStruct, Xt);
    
    error(i) = (sum((Group~=Yv))/length(Yv))*100;
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    n_sv(i) = length(SVMStruct.SupportVectors);
end

lbox=log10(bc);
figure();
subplot(2,1,1);
plot(lbox,error); hold on;
plot(lbox, errort);
grid on;
title('Error in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Error','interpreter','latex');
legend('Validation', 'Train');
subplot(2,1,2);
plot(lbox,n_sv,'r');
title('Number of Support Vectors in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
grid on;

% We chose order = 2 and BoxConstraint = 10^4

% Train SVM
tic
mdl = fitcsvm(Xtrain, Ytrain,'BoxConstraint',10^4,'KernelFunction','polynomial', 'PolynomialOrder',...
        2, 'Standardize', true, 'Solver', 'L1QP');
toc

labelv = predict(mdl, Xv);
valerror = (sum((labelv~=Yv))/length(Yv))*100;

% Performance
labelTest = predict(mdl, Xtest);
labelTrain = predict(mdl, Xtrain);
C1 = confusionmat(Ytest, labelTest);
C2 = confusionmat(Ytrain, labelTrain);
figure()
plotconfusion(Ytest', labelTest');
figure()
plotconfusion(Ytrain', labelTrain');


TP = C1(2,2); % True Positives
TN = C1(1,1); % True Negatives
FP = C1(1,2); % False Positives
FN = C1(2,1); % False Negatives

TP2 = C2(2,2); % True Positives
TN2 = C2(1,1); % True Negatives
FP2 = C2(1,2); % False Positives
FN2 = C2(2,1); % False Negatives


Accuracy = (TP+TN)/length(Ytest)*100;
Error = 100-Accuracy;
Sensitivity = (TP/(TP+FN))*100;
Specificity = (TN/(TN+FP))*100;
Precision = (TP/(TP+FP));
Recall = (TP/(TP+FN));
F_measure = 2*((Precision*Recall)/(Precision+Recall));

accuracyTrain = (TP2+TN2)/length(Ytrain)*100;
errorTrain = 100-accuracyTrain;

fprintf('Accuracy: %g%%\n', Accuracy);
fprintf('Error: %g%%\n', Error);
fprintf('Sensitivity: %g%%\n', Sensitivity);
fprintf('Specificity: %g%%\n', Specificity);
fprintf('Precision: %g\n', Precision);
fprintf('Recall: %g\n', Recall);
fprintf('F-measure: %g\n', F_measure);
fprintf('\nAccuracy Train: %g%%\n', accuracyTrain);
fprintf('Error Train: %g%%\n', errorTrain);

[Xp,Yp,Tp,AUC,Opto] = perfcurve(Ytest,labelTest,1);

figure()
plot(Xp,Yp, 'linewidth', 1.5); hold on;
plot(Opto(1), Opto(2), 'ro');
xlabel('False positive rate', 'interpreter', 'latex');
ylabel('True positive rate', 'interpreter', 'latex');
title('ROC for Classification by SVM (Polynomial) - Dataset 2', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;

ROC(4).data = [Xp Yp];
ROC(4).opto = Opto;
ROC(4).auc = AUC;

save('roc_auc2', 'ROC');

%% Decision Tree - dataset2.mat
clear all; close all; clc
load dataset2.mat
load roc_auc2

Xt = Xtrain(1:691,:);
Yt = Ytrain(1:691,:);
Xv = Xtrain(692:end,:);
Yv = Ytrain(692:end,:);

% Choice of Classifier Parameters
tree = fitctree(Xt, Yt); % original tree, without pruning

num_levels = max(tree.PruneList);

for i = 1:num_levels+1
    t = prune(tree, 'Level', i-1);
   
    labelTrain = predict(t, Xtrain);
    labelv = predict(t, Xv);
    
    errorTrain(i) = (sum(labelTrain~=Ytrain)/length(Ytrain))*100;
    errorVal(i) = (sum(labelv~=Yv)/length(Yv))*100;
end

figure()
plot(0:1:num_levels, errorTrain); hold on;
plot(0:1:num_levels, errorVal); hold on;
plot(0, errorTrain(1), 'ro');
title('Error for different Pruning Levels', 'interpreter', 'latex');
xlabel('Pruning Level', 'interpreter', 'latex');
ylabel('Error [\%]', 'interpreter', 'latex');
legend('Train', 'Validation', 'No Pruning', 'location', 'southeast');
grid on;

% We chose not to prune

% Training Decision Tree
tic
tree = fitctree(Xtrain, Ytrain);
tree = prune(tree, 'Level', 1);
toc
view(tree, 'Mode', 'Graph');

% Performance
labelTest = predict(tree, Xtest);
labelTrain = predict(tree, Xtrain);
C1 = confusionmat(Ytest, labelTest);
C2 = confusionmat(Ytrain, labelTrain);
figure()
plotconfusion(Ytest', labelTest');
figure()
plotconfusion(Ytrain', labelTrain');


TP = C1(2,2); % True Positives
TN = C1(1,1); % True Negatives
FP = C1(1,2); % False Positives
FN = C1(2,1); % False Negatives

TP2 = C2(2,2); % True Positives
TN2 = C2(1,1); % True Negatives
FP2 = C2(1,2); % False Positives
FN2 = C2(2,1); % False Negatives


Accuracy = (TP+TN)/length(Ytest)*100;
Error = 100-Accuracy;
Sensitivity = (TP/(TP+FN))*100;
Specificity = (TN/(TN+FP))*100;
Precision = (TP/(TP+FP));
Recall = (TP/(TP+FN));
F_measure = 2*((Precision*Recall)/(Precision+Recall));

accuracyTrain = (TP2+TN2)/length(Ytrain)*100;
errorTrain = 100-accuracyTrain;

fprintf('Accuracy: %g%%\n', Accuracy);
fprintf('Error: %g%%\n', Error);
fprintf('Sensitivity: %g%%\n', Sensitivity);
fprintf('Specificity: %g%%\n', Specificity);
fprintf('Precision: %g\n', Precision);
fprintf('Recall: %g\n', Recall);
fprintf('F-measure: %g\n', F_measure);
fprintf('\nAccuracy Train: %g%%\n', accuracyTrain);
fprintf('Error Train: %g%%\n', errorTrain);

[Xp,Yp,Tp,AUC,Opto] = perfcurve(Ytest,labelTest,1);

figure()
plot(Xp,Yp, 'linewidth', 1.5); hold on;
plot(Opto(1), Opto(2), 'ro');
xlabel('False positive rate', 'interpreter', 'latex');
ylabel('True positive rate', 'interpreter', 'latex');
title('ROC for Classification by Decision Tree - Dataset 2', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;

ROC(2).data = [Xp Yp];
ROC(2).opto = Opto;
ROC(2).auc = AUC;

save('roc_auc2', 'ROC');

%% Neural Network - dataset2.mat
close all; clear all; clc;
load dataset2.mat;
load roc_auc2

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
fprintf('Testing Error (mse): %g\n', tr. best_tperf);

% Performance
labelTest = y_test';
labelTrain = y_train';
C1 = [86 36;22 86];
C2 = [366 114;66 375];

TP = C1(2,2); % True Positives
TN = C1(1,1); % True Negatives
FP = C1(1,2); % False Positives
FN = C1(2,1); % False Negatives

TP2 = C2(2,2); % True Positives
TN2 = C2(1,1); % True Negatives
FP2 = C2(1,2); % False Positives
FN2 = C2(2,1); % False Negatives


Accuracy = (TP+TN)/length(Ytest)*100;
Error = 100-Accuracy;
Sensitivity = (TP/(TP+FN))*100;
Specificity = (TN/(TN+FP))*100;
Precision = (TP/(TP+FP));
Recall = (TP/(TP+FN));
F_measure = 2*((Precision*Recall)/(Precision+Recall));

accuracyTrain = (TP2+TN2)/length(Ytrain)*100;
errorTrain = 100-accuracyTrain;

fprintf('Accuracy: %g%%\n', Accuracy);
fprintf('Error: %g%%\n', Error);
fprintf('Sensitivity: %g%%\n', Sensitivity);
fprintf('Specificity: %g%%\n', Specificity);
fprintf('Precision: %g\n', Precision);
fprintf('Recall: %g\n', Recall);
fprintf('F-measure: %g\n', F_measure);
fprintf('\nAccuracy Train: %g%%\n', accuracyTrain);
fprintf('Error Train: %g%%\n', errorTrain);

[Xp,Yp,Tp,AUC,Opto] = perfcurve(Ytest,labelTest,1);

figure()
plot(Xp,Yp, 'linewidth', 1.5); hold on;
plot(Opto(1), Opto(2), 'ro');
xlabel('False positive rate', 'interpreter', 'latex');
ylabel('True positive rate', 'interpreter', 'latex');
title('ROC for Classification by Neural Network - Dataset 2', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;

ROC(3).data = [Xp Yp];
ROC(3).opto = Opto;
ROC(3).auc = AUC;

save('roc_auc2', 'ROC');

%% ROC & AUC for all classifiers - dataset2.mat
clear all; close all; clc;
load roc_auc2

figure()
for i = 1:length(ROC)
    plot(ROC(i).data(:,1),ROC(i).data(:,2), 'linewidth', 1.5); hold on;
    h(i) = plot(ROC(i).opto(1), ROC(i).opto(2), 'ro');
    xlabel('False positive rate', 'interpreter', 'latex');
    ylabel('True positive rate', 'interpreter', 'latex');
    title('ROC for Classification by All Classifiers - Dataset 2', 'interpreter', 'latex');
    grid on; hold on;
end
set(get(get(h(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h(3),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend(sprintf('SVM (RBF): AUC = %.3f',ROC(1).auc), sprintf('Decision Tree: AUC = %.3f',ROC(2).auc),...
       sprintf('Neural Network: AUC = %.3f',ROC(3).auc), ...
       sprintf('SVM (poly.): AUC = %.3f',ROC(4).auc),'Location', 'Southeast');
   
   
%% 
clear all; close all; clc
load dataset2.mat
load roc_auc2
load Feature_Select2

% Feature Selection
Xtrain = Xtrain(:, inmodel);
Xtest = Xtest(:, inmodel);

% Choice of Classifier Parameters
s_test=0.1:0.05:2.5;
i = 1;

    SVMStruct = fitcsvm(Xtrain, Ytrain,'BoxConstraint',1,'KernelFunction','polynomial', 'PolynomialOrder',...
        4, 'Standardize', true, 'Solver', 'L1QP');
    
    Group = predict(SVMStruct, Xtest);
    label = predict(SVMStruct, Xtrain);
    
    error(i) = (sum((Group~=Ytest))/length(Ytest))*100;
    errort(i) = (sum((label~=Ytrain))/length(Ytrain))*100;
    
    n_sv(i) = length(SVMStruct.SupportVectors);

