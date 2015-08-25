function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[dim_x, dim_y] = size(z);

if (dim_x == 1) & (dim_y ==1)
   g=1/(1+exp(-z));      
else

   g=1+exp(-z);
   
   for m=1:dim_x
    for n=1:dim_y
     g(m,n)=1/g(m,n);
    end
   end
   
end


% =============================================================

end
