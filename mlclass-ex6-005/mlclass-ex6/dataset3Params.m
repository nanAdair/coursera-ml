function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

limit = 100;

CTest = 0.01;
sigmaTest = 0.01;

x1 = X(1,:);
x2 = X(2,:);

model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
preError = mean(double(predictions ~= yval));
res = preError;

cv = [];
sv = [];

base = 0.01;

while base < limit,
    cv = [cv; base];
    sv = [sv; base];
    base *= 3;
end

sizeCV = size(cv, 1);
sizeSV = size(sv, 1);

for i = 1 : sizeCV,
    for j = 1 : sizeSV,
        CTest = cv(i);
        sigmaTest = sv(j);
        model = svmTrain(X, y, CTest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
        predictions = svmPredict(model, Xval);
        preError = mean(double(predictions ~= yval));
        
        if preError < res,
            res = preError;
            C = CTest;
            sigma = sigmaTest;
        end
        
    end
end




% =========================================================================

end
