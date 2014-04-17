function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Forward Propagation

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = 1 ./ (1 + exp(-z2));

a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = 1 ./ (1 + exp(-z3));

% Generate the y value matrices
right = zeros(m, num_labels);
for i = 1 : m,
    for j = 1 : num_labels,
        if(y(i) == j),
            right(i, j) = 1;
        end
    end
end

res = 0;
for i = 1 : m,
    for j = 1 : num_labels,
        res += right(i, j) * log(a3(i, j)) + (1 - right(i, j)) * log(1 - a3(i, j));
    end
end

J = -res / m;

% Add regularization

temp = 0;
for i = 1: size(Theta1, 1),
    for j = 2: size(Theta1, 2),
        temp += Theta1(i, j) ^ 2;
    end
end

for i = 1: size(Theta2, 1),
    for j = 2: size(Theta2, 2),
        temp += Theta2(i, j) ^ 2;
    end
end

temp = temp * lambda / (2 * m);

J = J + temp;
        
% J value is done
% Backpropagation

bigDelta1 = zeros(size(Theta1));
bigDelta2 = zeros(size(Theta2));
            
for i = 1 : m,
    a1 = X(i, :);
    a1 = [1 a1];
    
    z2 = a1 * Theta1';
    a2 = 1 ./ (1 + exp(-z2));

    a2 = [1 a2];
    z3 = a2 * Theta2';
    a3 = 1 ./ (1 + exp(-z3));
    
    %size(a3)
    %size(right(i, :))
    
    % !!!!!!!!!!!
    % Pay attention to the :, when you have to get one line from the matrices
    % !!!!!!!!!!!
    delta3 = a3 - right(i, :);
    
    delta2 = delta3 * Theta2 .* a2 .* (1 - a2);
    delta2 = delta2(2 : end);
    
    %for j = 1 : size(Theta1, 1),
        %for k = 1 : size(Theta1, 2),
            %bigDelta1(j, k) = bigDelta1(j, k) + delta2(j) * a1(k);
        %end
    %end
    
    %for j = 1 : size(Theta2, 1),
        %for k = 1 : size(Theta2, 2),
            %bigDelta2(j, k) = bigDelta2(j, k) + delta3(j) * a2(k);
        %end
    %end
    bigDelta1 = bigDelta1 + delta2' * a1; 
    bigDelta2 = bigDelta2 + delta3' * a2;
    
end

reg1 = zeros(size(Theta1));
reg2 = zeros(size(Theta2));

reg1(:, 2 : end) = Theta1(:, 2 : end);
reg2(:, 2 : end) = Theta2(:, 2 : end);
    
Theta1_grad = bigDelta1 / m + lambda * reg1 / m;
Theta2_grad = bigDelta2 / m + lambda * reg2 / m;



            

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
