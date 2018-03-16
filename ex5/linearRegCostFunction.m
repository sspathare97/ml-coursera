function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;

J = sum((h-y).^2)/(2*m) + lambda*(theta'*theta-theta(1)^2)/(2*m);

% '(h - y).*X'
% % (h - y).*X
% 'size(X)'
% size(X)
% 'size(theta)'
% size(theta)
% 'sum((h - y).*X,1)'
% sum((h - y).*X,1)
% % '(h - y).*X(:,1)'
% % (h - y).*X(:,1)
% 'sum((h - y).*X(:,1),1)'
% sum((h - y).*X(:,1),1)
% '1'
% grad = sum((h - y).*X)'/m + lambda*theta/m
% '1'
grad = sum((h - y).*X)'/m + lambda*([ 0; theta(2:length(theta))])/m;
% '2'
% grad(1) = sum(h - y)/m
% grad(1) = 12345;


% =========================================================================

grad = grad(:);

end
