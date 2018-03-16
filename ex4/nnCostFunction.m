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
% X 5000 x 400
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
X = [ones(m, 1) X];
a1 = X;
% 5000,401
z2 = a1*Theta1';
% 	5000,401 * 401,25 = 5000,25
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
% 5000, 26
z3 = a2*Theta2';
% 	5000,26 * 26,10 = 5000,10
a3 = sigmoid(z3);
h = a3;

K = size(h,2);

ty = y;
y = zeros(m,K);
for i=1:m
	y(i,ty(i))=1;
end

J = 0;
for i=1:m
	for k=1:K
		J += ( -y(i,k)*log(h(i,k)) -(1-y(i,k))*log(1-h(i,k)) );
	end
end
J = J/m;


% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
p = size(a1,2) -1; 	%400
q = size(a2,2) -1; 	%25
r = size(h,2);		%10
tmp = 0;
for j=1:q
	for k=2:p+1
		tmp += Theta1(j,k)*Theta1(j,k);
	end
end
for j=1:r
	for k=2:q+1
		tmp += Theta2(j,k)*Theta2(j,k);
	end
end

J += lambda*tmp/(2*m);


%-------------------------------------------------------------
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));


% p = size(a1,2) -1; 	%400
% q = size(a2,2) -1; 	%25
% r = size(h,2);		%10
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% Theta1_grad = zeros(q,p+1);
% Theta2_grad = zeros(r,q+1);

for t = 1:m

	% x = X(t,:);
	% a1 = x;
	% % 'a1'
	% % size(a1)
	% % 1,401
	% z2 = a1*Theta1';
	% % 1,401 * 401,25 = 1,25
	% a2 = sigmoid(z2);
	% a2 = [1 a2];
	% % 'a2'
	% % size(a2)
	% % 1,26
	% z3 = a2*Theta2';
	% % 1,26 * 26,10 = 1,10
	% a3 = sigmoid(z3);
	% % 'a3'
	% % size(a3)
	% % 1,10
	% h = a3;
	% K = size(h,2);

	d3 = a3(t,:) - y(t,:);
	% 1,10

	% Theta1 has size 25 x 401
	% Theta2 has size 10 x 26

	% tm1 = d3*Theta2;
	% % 1,10*10,26 = 1,26
	% tm1 = tm1(2:end);
	% % 1,25
	% d2 = tm1 .* sigmoidGradient(z2);
	% % 1,25 .* 1,25 = 1,25

	% d2 = d3*Theta2 .* a2(t,:) .* (1-a2(t,:));
	% % 1,10*10,26 .* 1,26 = 1,26
	% % Theta2' * d3'
	% % 26,10 * 10,1
	% d2 = d2(2:end);
	% % 1,25

	% % imp
	% Theta1_grad += d2'*a1(t,:);
	% 			% 25,1* 1,401 = 25,401
	% Theta2_grad += d3'*a2(t,:);
	% 			% 10,1* 1,26 = 10,26

	d2 = Theta2' * d3' .* sigmoidGradient([1, z2(t, :)])';
    d2 = d2(2:end);

    Theta1_grad = Theta1_grad + d2 * a1(t, :);
    Theta2_grad = Theta2_grad + d3' * a2(t, :);


end

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% imp
Theta1_grad /= m;
Theta2_grad /= m;


% -------------------------------------------------------------


Theta1_grad += lambda * [ zeros(q,1) Theta1(:,2:end) ] /m;
Theta2_grad += lambda * [ zeros(r,1) Theta2(:,2:end) ] /m;




% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
