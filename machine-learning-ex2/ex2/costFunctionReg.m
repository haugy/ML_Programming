function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Z input of g(z)
Z = X * theta;

%A out of g(Z)
A = 1 ./ (1 + exp(-Z));

%J value
J = 1/m * sum(-y.*log(A) - (1 - y).*log(1 - A))...
  + lambda/(2*m) * sum(theta(2:size(theta, 1)).^2);

%L is a matrix with 0 at the top left and 1's down the diagonal, 
%                 with 0's everywhere else.
L = eye(size(theta, 1));
L(1, 1) = 0;
  
%grad value
grad = 1/m * ((A - y)'*X)' .+ (lambda / m * L * theta);
%grad(2:size(grad, 1),:) = grad(2:size(grad, 1),:) ...
%                        .+ (lambda / m * theta(2:size(theta, 1),:));


% =============================================================

end
