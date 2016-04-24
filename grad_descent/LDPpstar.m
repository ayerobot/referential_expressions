function lambda = LDPpstar(hx, Tx_all, theta, accuracy, lambda)
% Dimensions:
%    n => number of grid points
%    c => number of features
%    s => number of commands
    
% Inputs:
%    hx => an (n, 1)-dimensional probability vector
%    Tx => an (n, c)-dimensional vector of sufficient statistics
%    theta => a (c, 1)-dimensional constraint vector
%    accuracy => level of accuracy (e.g. 1e-3)
    
%    Optional Arguments
%        lambda => initialized value of lambda
%    
% Variables:
%    lambda => a (c, 1)-dimensional weight vector
%    zlambda => a (s, 1)-dimensional vector of normalization constants,
%               one for every command (because each command produces a 
%               different distribution)
%    expvals => a (c, 1)-dimensional vector giving the expected value of
%               the sufficient statistics across all commands for a given 
%               lambda. Goal is to get this to be close to theta.

% If no initialization for lambda, initialize to zero.
if nargin < 5
    c = size(Tx_all{1}, 2);
    lambda = zeros(c, 1);
end

zlambda = get_zlambda(hx, lambda, Tx_all);
expvals = get_expvals(hx, lambda, zlambda, Tx_all);
epsilon = 0.01; % step size

% While L2 norm between expected value of features across all commands for
% a given lambda and true average value is greater than accuracy,
while (norm(expvals - theta) > accuracy)
    disp(['Error = ' num2str(norm(expvals - theta))])
    lambda = lambda - epsilon*(expvals - theta); % grad descent
    zlambda = get_zlambda(hx, lambda, Tx_all); % recalculate all normalization constants
    expvals = get_expvals(hx, lambda, zlambda, Tx_all); % recalculate expected values
end
end

function expvals = get_expvals(hx, lambda, zlambda, Tx_all)
% Calculates expected values for a given lambda
c = size(Tx_all{1}, 2);
expvals = zeros(c, 1);
num_commands = length(Tx_all);
for i = 1:num_commands
    expvals = expvals + sum(Tx_all{i}.*repmat(hx.*exp(Tx_all{i}*lambda)/zlambda(i), 1, c))';
end
expvals = expvals/num_commands;
end

function zlambda = get_zlambda(hx, lambda, Tx_all)
% Calculates normalization constant for all commands for a given lambda
num_commands = length(Tx_all);
zlambda = zeros(num_commands, 1);
for i = 1:num_commands
    zlambda(i) = sum(exp(Tx_all{i}*lambda).*hx);
end
end

