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
one = ones(m, 1);

%Make a vector to represent lambda
factor = ones(size(theta));
factor(1) = 0;
for i = 2: size(theta)(1),
    factor(i) = lambda;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = -(1 / m) * (log(sigmoid(X * theta))' * y + log(1 - sigmoid(X * theta))' * (one - y)) + (sum(theta .^2) - theta(1).^2) * lambda / (2 * m);

% seperately update the grad value;
%grad(1) = grad(1) - (X(:, 1)' * (y - sigmoid(X * theta))) / m;
%for i = 2: m,
    %grad(i) = grad(i) - (X(:, i)' * (y - sigmoid(X * theta))) / m + lambda * theta(i) / m;
%end

% update the value using vectorized method
grad = grad - (X' * (y - sigmoid(X * theta))) / m + factor .* theta / m;

% =============================================================

end
