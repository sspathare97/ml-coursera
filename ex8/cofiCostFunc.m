function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% for i=1:num_movies
% 	for j=1:num_users
% 		% if(R(i,j) == 1)
% 			% Theta(j,:)
% 			% X(i,:)
% 			% Y(i,j)
% 			J += R(i,j)*((Theta(j,:)*X(i,:)' - Y(i,j))^2);
% 			% pause;
% 		% end
% 	end
% end

% % for k=1:num_features
% % 	for i=1:num_movies
% % 		for j=1:num_users
% % 			X_grad(i,k) += R(i,j)*Theta(j,k)*(X(i,:)*Theta(j,:)' - Y(i,j));
% % 			Theta_grad(j,k) += R(i,j)*X(i,k)*(X(i,:)*Theta(j,:)' - Y(i,j));
% % 		end
% % 	end
% % end

% % Notes: X - num_movies  x num_features matrix of movie features
% %        Theta - num_users  x num_features matrix of user features
% %        Y - num_movies x num_users matrix of user ratings of movies
% %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
% %            i-th movie was rated by the j-th user
% %        X_grad - num_movies x num_features matrix, containing the 
% %                 partial derivatives w.r.t. to each element of X
% %        Theta_grad - num_users x num_features matrix, containing the 
% %                     partial derivatives w.r.t. to each element of Theta
% % num_features 3
% % num_movies 5
% % num_users 4

% % Notes: X - 5  x 3
% %        Theta - 4  x 3
% %        Y - 5 x 4
% %        R - 5 x 4
% %        X_grad - 5 x 3
% %        Theta_grad - 4 x 3 

% for i=1:num_movies
% 	% 3,4 * 1,4
% 	X_grad += sum(sum(Theta' .* (R(i,:).*(X(i,:)*Theta' - Y(i,:)))));
% 	% 					1,4 .* 1,4 .* 1,4
% 	% X_grad(i,k) += sum(Theta(:,k)'.*R(i,:).*(X(i,:)*Theta' - Y(i,:)));

% 	for k=1:num_features
		
% 		% for j=1:num_users
% 		% 	% X_grad(i,k) += R(i,j)*Theta(j,k)*(Theta*X(i,:)' - Y(i,:));
% 		% end

% 		% Theta 4,3
% 		% X(i,:) 1,3
% 		% Theta * X(i,:)'' 4,3 * 3,1 = 4,1
% 		% X(i,:)*Theta' - Y(i,:)
% 		% Theta(:,k)'.*(X(i,:)*Theta' - Y(i,:))
% 		% R(i,:).*Theta(:,k)'.*(X(i,:)*Theta' - Y(i,:))
% 		% sum(R(i,:).*Theta(:,k)'.*(X(i,:)*Theta' - Y(i,:)))

% 		% X_grad(i,k) += sum(R(i,:).*Theta(:,k)'.*(X(i,:)*Theta' - Y(i,:)));


% 	end
% end

% for k=1:num_features
% 	for i=1:num_movies
% 		for j=1:num_users
% 			Theta_grad(j,k) += R(i,j)*X(i,k)*(X(i,:)*Theta(j,:)' - Y(i,j));
% 		end
% 	end
% end

% J = sum(sum(R.*((X*Theta' - Y).^2)))/2;
% calculating cost function.
df = (X*Theta'-Y);
J = sum((df.^2)(R==1))/2;
J = J + lambda*sum(sum(Theta.^2))/2;  % regularized term of theta.
J = J + lambda*sum(sum(X.^2))/2;     % regularized term of x.

X_grad = (df.*R)*Theta;                 %unregularized vectorized implementation
Theta_grad = ((df.*R)'*X);              %unregularized vectorized implementation


X_grad = X_grad + (lambda * X);             % regularized
Theta_grad = Theta_grad + (lambda * Theta);  % regularized









% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
