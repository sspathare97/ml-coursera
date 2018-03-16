function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%
% m = size(X, 1);

% % You need to return the following variables correctly.
% Z = zeros(m, K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%                    projection_k = X(i, :) * U(:, k);
%
% xsz = size(X); 50 2
% usz = size(U); 2 2
% ursz = size(Ur); 2 K

% Ur = U(:,1:K);
% Z = Ur'*X = (2,K)'*(50,2) = (K,2)*(2,50) = K,50

% Z = X*Ur = 50,2 * 2,K ;
Z = X*U(:,1:K);
% =============================================================

end
