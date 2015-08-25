function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X * theta;

y1 = h - y;

J = (y1' * y1)/(2*m)

string = [' theta_0=',num2str(theta(1)),', theta_1=',num2str(theta(2)),', J=', num2str(J)];

%disp (string)

% =========================================================================

end
