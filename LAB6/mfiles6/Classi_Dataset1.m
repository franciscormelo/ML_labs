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

%% Dataset 1

%% Naive Bayes - dataset1.mat
close all; clear all; clc;
load dataset1.mat

% Training Naive Bayes
tic
mdl = fitcnb(Xtrain, Ytrain);
toc

% Performance
label = predict(mdl, Xtest);
labelTrain = predict(mdl, Xtrain);
C1 = confusionmat(Ytest, label);
C2 = confusionmat(Ytrain, labelTrain);
figure()
plotconfusion(Ytest', label');
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

[Xp,Yp,Tp,AUC, Opto] = perfcurve(Ytest,label,1);

figure()
plot(Xp,Yp, 'linewidth', 1.5); hold on;
plot(Opto(1), Opto(2), 'ro');
xlabel('False positive rate', 'interpreter', 'latex');
ylabel('True positive rate', 'interpreter', 'latex');
title('ROC for Classification by Naive Bayes - Dataset 1', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;

ROC(1).data = [Xp Yp];
ROC(1).opto = Opto;
ROC(1).auc = AUC;

m = matfile('roc_auc.mat','Writable',true);
save('roc_auc', 'ROC');

%% Support Vector Machine (RBF) - dataset1.mat
clear all; close all; clc
load dataset1.mat
load idc
load roc_auc

%idx = randperm(93);
%save('idc', 'idx');

Xt = [Xtrain(idx(1:30),:);Xtrain(idx(55:93),:)];
Yt = [Ytrain(idx(1:30),:);Ytrain(idx(55:93),:)];
Xv = Xtrain(idx(31:54),:);
Yv = Ytrain(idx(31:54),:);

% Choice of Classifier Parameters
s_test=0.1:0.05:10;

for i=1:length(s_test)
    SVMStruct = fitcsvm(Xt, Yt,'BoxConstraint',10,'KernelFunction','RBF', 'KernelScale',...
        s_test(i), 'Standardize', true, 'Solver', 'L1QP');
    
    label = predict(SVMStruct, Xt);
    labelv = predict(SVMStruct, Xv);
    
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    errorv(i) = (sum((labelv~=Yv))/length(Yv))*100;
    
    n_sv(i) = length(SVMStruct.SupportVectors);
end

figure();
subplot(2,1,1);
plot(s_test, errort); hold on;
plot(s_test, errorv);
grid on;
title('Error in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Error','interpreter','latex');
xlim([0 5]);
legend('Train', 'Validation');
subplot(2,1,2);
plot(s_test,n_sv,'r');
title('Number of Support Vectors in respect to $\sigma$','interpreter','latex');
xlabel('$\sigma$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
xlim([0 5]);
grid on;

clear error;
clear errort;
clear errorv;
clear n_sv;

bc=[10^-4 10^-3 10^-2 10^-1 10^0 10^0.5 10^1 10^2 10^3 10^4];

for i=1:length(bc)
    SVMStruct = fitcsvm(Xt, Yt,'BoxConstraint',bc(i),'KernelFunction','RBF', 'KernelScale',...
        2, 'Standardize', true, 'Solver', 'L1QP');
   
    label = predict(SVMStruct, Xt);
    labelv = predict(SVMStruct, Xv);
    
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    errorv(i) = (sum((labelv~=Yv))/length(Yv))*100;
    
    n_sv(i) = length(SVMStruct.SupportVectors);
end

lbox=log10(bc);
figure();
subplot(2,1,1);
plot(lbox, errort); hold on;
plot(lbox, errorv);
grid on;
title('Error in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Error','interpreter','latex');
legend('Train', 'Validation');
subplot(2,1,2);
plot(lbox,n_sv,'r');
title('Number of Support Vectors in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
grid on;

% We chose sigma = 2 and BoxConstraint = 10


% Train SVM
tic
mdl = fitcsvm(Xtrain, Ytrain,'BoxConstraint',10,'KernelFunction','RBF','KernelScale', 2,'Standardize', true, 'Solver', 'L1QP');
toc


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
title('ROC for Classification by SVM (RBF) - Dataset 1', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;

ROC(2).data = [Xp Yp];
ROC(2).opto = Opto;
ROC(2).auc = AUC;

save('roc_auc', 'ROC');

%% Support Vector Machine (Polynomial) - dataset1.mat
clear all; close all; clc
load dataset1.mat
load idc
load roc_auc

Xt = [Xtrain(idx(1:30),:);Xtrain(idx(55:93),:)];
Yt = [Ytrain(idx(1:30),:);Ytrain(idx(55:93),:)];
Xv = Xtrain(idx(31:54),:);
Yv = Ytrain(idx(31:54),:);

% Choice of Classifier Parameters
s_test=1:1:10;

for i=1:length(s_test)
    SVMStruct = fitcsvm(Xt, Yt,'BoxConstraint',10,'KernelFunction','polynomial', 'PolynomialOrder',...
        s_test(i), 'Standardize', true, 'Solver', 'L1QP');
    
    label = predict(SVMStruct, Xt);
    labelv = predict(SVMStruct, Xv);
    
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    errorv(i) = (sum((labelv~=Yv))/length(Yv))*100;
    
    n_sv(i) = length(SVMStruct.SupportVectors);
end

figure();
subplot(2,1,1);
plot(s_test, errort); hold on;
plot(s_test, errorv);
grid on;
title('Error in respect to $p$','interpreter','latex');
xlabel('$p$ Order','interpreter','latex');
ylabel('Error','interpreter','latex');
xlim([1 10]);
legend('Train', 'Validation');
subplot(2,1,2);
plot(s_test,n_sv,'r');
title('Number of Support Vectors in respect to $\sigma$','interpreter','latex');
xlabel('$p$ Order','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
xlim([1 10]);
grid on;

clear error;
clear errort;
clear errorv;
clear n_sv;

bc=[10^-4 10^-3 10^-2 10^-1 10^0 10^0.25 10^0.5 10^1 10^2 10^3 10^4];

for i=1:length(bc)
    SVMStruct = fitcsvm(Xt, Yt,'BoxConstraint',bc(i),'KernelFunction','polynomial', 'PolynomialOrder',...
        2, 'Standardize', true, 'Solver', 'L1QP');
    
    label = predict(SVMStruct, Xt);
    labelv = predict(SVMStruct, Xv);
    
    errort(i) = (sum((label~=Yt))/length(Yt))*100;
    errorv(i) = (sum((labelv~=Yv))/length(Yv))*100;
    
    n_sv(i) = length(SVMStruct.SupportVectors);
end

lbox=log10(bc);
figure();
subplot(2,1,1);
plot(lbox, errort); hold on;
plot(lbox, errorv);
grid on;
title('Error in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Error','interpreter','latex');
legend('Train', 'Validation');
subplot(2,1,2);
plot(lbox,n_sv,'r');
title('Number of Support Vectors in respect to boxconstraint','interpreter','latex');
xlabel('$log(boxconstraint)$','interpreter','latex');
ylabel('Number of Support Vectors','interpreter','latex');
grid on;

% We chose polynomial order = 2 and BoxConstraint = 10


% Train SVM
tic
mdl = fitcsvm(Xtrain, Ytrain,'BoxConstraint',10,'KernelFunction','polynomial', 'PolynomialOrder',...
        2, 'Standardize', true, 'Solver', 'L1QP');
toc


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
title('ROC for Classification by SVM (Polynomial) - Dataset 1', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f', AUC), 'Location', 'Southeast');
grid on;

ROC(4).data = [Xp Yp];
ROC(4).opto = Opto;
ROC(4).auc = AUC;

save('roc_auc', 'ROC');

%% Decision Tree - dataset1.mat
clear all; close all; clc
load dataset1.mat
load roc_auc

% Choice of Classifier Parameters
tree = fitctree(Xtrain, Ytrain); % original tree, without pruning

num_levels = max(tree.PruneList);

for i = 1:num_levels+1
    t = prune(tree, 'Level', i-1);
    
    labelTrain = predict(t, Xtrain);
    
    errorTrain(i) = (sum(labelTrain~=Ytrain)/length(Ytrain))*100;
end

figure()
plot(0:1:num_levels, errorTrain); hold on;
plot(0, errorTrain(1), 'ro');
title('Error for different Pruning Levels', 'interpreter', 'latex');
xlabel('Pruning Level', 'interpreter', 'latex');
ylabel('Error [\%]', 'interpreter', 'latex');
legend('Train', 'No Pruning', 'location', 'southeast');
grid on;

% We chose not to prune the tree

% Training Decision Tree
tic
tree = fitctree(Xtrain, Ytrain);

toc

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
title('ROC for Classification by Decision Tree - Dataset 1', 'interpreter', 'latex');
legend(sprintf('AUC = %.3f',AUC), 'Location', 'Southeast');
grid on;

ROC(3).data = [Xp Yp];
ROC(3).opto = Opto;
ROC(3).auc = AUC;

save('roc_auc', 'ROC');

%% ROC & AUC for all classifiers - dataset1.mat
clear all; close all; clc;
load roc_auc

figure()
for i = 1:length(ROC)
    plot(ROC(i).data(:,1),ROC(i).data(:,2), 'linewidth', 1.5); hold on;
    h(i) = plot(ROC(i).opto(1), ROC(i).opto(2), 'ro');
    xlabel('False positive rate', 'interpreter', 'latex');
    ylabel('True positive rate', 'interpreter', 'latex');
    title('ROC for Classification by All Classifiers - Dataset 1', 'interpreter', 'latex');
    grid on; hold on;
end
set(get(get(h(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h(3),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend(sprintf('Naive Bayes: AUC = %.3f',ROC(1).auc), sprintf('SVM (RBF): AUC = %.3f',ROC(2).auc),...
       sprintf('Decision Tree: AUC = %.3f',ROC(3).auc),...
       sprintf('SVM (poly.): AUC = %.3f',ROC(4).auc),'Location', 'Southeast');
