function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

figure;

col='rgb';
clen=length(col);
colc=1;

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
      t = [theta0_vals(i); theta1_vals(j)];
      J_vals(i,j) = computeCost(X, y, t);
    end
end
J_vals = J_vals';
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20));
xlabel('\theta_0'); ylabel('\theta_1');
hold on;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    temp1=theta(1)-sum((theta(1)+theta(2)*X(:,2)-y))*alpha/m;
    temp2=theta(2)-sum((theta(1)+theta(2)*X(:,2)-y).*X(:,2))*alpha/m;
    theta(1)=temp1;
    theta(2)=temp2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    if(mod(iter,100)==0)
        ln = strcat(col(colc),'x');
        plot(theta(1), theta(2), ln, 'MarkerSize', 10, 'LineWidth', 2);
        colc++;
        if(colc>clen)
            colc=1;
        endif
    endif
end
ln = strcat(col(colc),'x');
plot(theta(1), theta(2), ln, 'MarkerSize', 10, 'LineWidth', 2);
hold off;
end
