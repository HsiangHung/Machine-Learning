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

%size(Theta1) %% = 25 x 401
%size(Theta2) %% = 10 x 26


X2 = [ones(m,1) X];

Y = zeros(m,num_labels);
yk= zeros(num_labels,1);
%size(Y)


for i = 1 : m
  Y(i,y(i))=1;
end


DELTA_2 = zeros(size(Theta2_grad));
DELTA_1 = zeros(size(Theta1_grad));


%size(DELTA_1)
%size(DELTA_2)

%pause

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


a1 = zeros(size(X2,2),1);

a22= zeros(hidden_layer_size,1);
a11= zeros(input_layer_size,1);
delta_22 =zeros(hidden_layer_size,1);

%size(a1)

for i = 1 : m

   yk(:) = Y(i,:);

   a1 = X2(i,:)';
   z2 = Theta1 * a1;
   a2 = [ones(1,1); sigmoid(z2)];
   
   z3 = Theta2 * a2;
   a3 = sigmoid(z3);
   h=a3;

   J = J + yk' * log(h) + (1-yk)' * log(1-h);

   % =============================================
   % The following is used for backprop:
   
   % step 2:
   
   delta_3 = a3 - yk;                                           % delta_3 is 10x1
      
   % step 3:
   
   delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);  % Theta2 =10x26
                                                               % delta_3 = 10x1
                                                               % [1;z2] = 26x1
                                                               % then delta_2 = 26x1
   % step 4:
      
    %for j=1:num_labels
    %  for k=1:hidden_layer_size
    %     
    %   DELTA_2(j,k) = DELTA_2(j,k) + delta_3(j)*a2(k+1);       % delta_3=10x1
    %                                                           % remove a2(1), bias unit
    %  end
    %end
    
    %a22(1:hidden_layer_size) = a2(2:hidden_layer_size+1); 
    
    DELTA_2 = DELTA_2 + kron(delta_3,a2');
     
    % ------------------------------------
       
    %for j=1:hidden_layer_size
    %  for k=1:input_layer_size
    %     
    %   DELTA_1(j,k) = DELTA_1(j,k) + delta_2(j+1)*a1(k+1);     % delta_2=26x1 
    %                                                           % remove delta_2(1) & a1(1)
    %  end
    %end
    
    %delta_22(1:hidden_layer_size) = delta_2(2:hidden_layer_size+1);
    %a11(1:input_layer_size) = a1(2:input_layer_size+1);
    
    
    delta_2 = delta_2(2:hidden_layer_size+1);
    
    DELTA_1 = DELTA_1 + kron(delta_2,a1');
   
   % ==============================================
   
end

%size(delta_2)
%z2
%size(DELTA_2)
%size(DELTA_1)
%[1;z2]
%sigmoidGradient([1;z2])
%C = kron(delta_22,a11');
%size(C)





J = -J/m;


%%  ===========================================================
%%  now add the regularization terms in the cost function:

J_reg=0;
for i = 1 : hidden_layer_size
  for j = 2 : input_layer_size+1
    J_reg = J_reg + Theta1(i,j)**2;
  end
end

for i = 1 : num_labels
  for j = 2 : hidden_layer_size+1
    J_reg = J_reg + Theta2(i,j)**2;
  end
end

J = J + lambda * J_reg/(2*m);







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





Theta1_grad = DELTA_1 /m;
Theta2_grad = DELTA_2 /m;












% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%m1=size(Theta1_grad,1)
n1=size(Theta1_grad,2);

%m2=size(Theta2_grad,1)
n2=size(Theta2_grad,2);


Theta1_grad(:,2:n1) = Theta1_grad(:,2:n1) + lambda*Theta1(:,2:n1)/m;
Theta2_grad(:,2:n2) = Theta2_grad(:,2:n2) + lambda*Theta2(:,2:n2)/m;


%size(Theta1_grad)
%size(Theta2_grad)












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
