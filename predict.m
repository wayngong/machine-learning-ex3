function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m,1) X]; % X size is m x 401
% Theta1 size is 25 x 401
% Theta2 size is 10 x 26
% a2 = g(z2) = g(theta1*a1)
a2_temp = sigmoid(Theta1*X'); % a2_temp size is 25 x m
a2 = [ones(m, 1), a2_temp']; % a2 size is m x 26
a3 = sigmoid(Theta2*a2'); % a3 size is 10 x m
[max, p] = max(a3', [], 2); % p size is m x 1





% =========================================================================


end
