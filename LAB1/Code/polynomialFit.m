%% polynomialFit
% Function to fit a polynomial of degree P to 1D data varaibles x and y
%
% Inputs: x,y - Data varaiables 
%         p - polynomial order
%
% Outputs: beta - Parameters estimates 
%          ypred - training data

function [beta,ypred] =polynomialFit(x,y,p)

sizex=size(x);

%Creation of design matrix
X= ones(sizex(1,1),p+1);

for i = 1:p
    X(:,i+1) = x.^i;
end

K=X'*X;
detK=det(K);

% Check if X'*X is invertible
if detK == 0
    fprintf('Matrix X^T*X is non invertible');
      
else
    beta=inv((X'*X))*X'*y; %Parameters estimates
    
    ypred=X*beta; %Calculation of the prediction(training data)
end

end