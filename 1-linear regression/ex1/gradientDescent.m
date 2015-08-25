function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h = X * theta;
    y1 = h - y;
    % here to compute the vector y1(i)=(h(x(i))-y(i)).
    J1=(y1' * y1)/(2*m);

    
    y0 = (X(:,1)' * y1)/m;
    y1 = (X(:,2)' * y1)/m;
    gradient = [y0, y1];
    % compute the gradient vector d(J)/d(theta_j), where
    % d(J)/d(theta_j) = \sum_i (h(x(i))-y(i))/m
    
    
    theta = theta - alpha * gradient';
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
