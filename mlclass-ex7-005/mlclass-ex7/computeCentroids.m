function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% Add by wbn, used for vectorised computing
% 注意多从矩阵的size来考虑如何进行矩阵化的运算
auxMatrix = zeros(m, K);
countNumber = zeros(K, 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i = 1 : size(idx, 1),
    auxMatrix(i, idx(i)) = 1;
    countNumber(idx(i)) += 1;
end

centroids = auxMatrix' * X;

for i = 1 : K,
    if countNumber(i) != 0,
        centroids(i, :) /= countNumber(i);
    end
end



% =============================================================


end

