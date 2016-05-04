% Tx_all => size 12 cell array. Each cell contains the value of the
%           features on the discretization of the board for a given command.
% all_theta => average value of features at every data point (over all commands).
% hx => uniform distribution over all data points
% 

num_features = 2;
num_commands = 12;

Tx_all = {};
all_theta = zeros(num_features, 1);
for i = 1:num_commands
    load(['command' num2str(i) '.mat'])
    Tx_all{i} = Tx;
    all_theta = all_theta + theta';
end
all_theta = all_theta/num_commands;
hx = ones(length(Tx_all{1}), 1)/length(Tx_all{1});

% if you preset a lambda, it won't be overwritten. This is helpful b/c you
% can initialize lambda to some value so you don't always have to start at
% the beginning.
if (~exist('lambda', 'var') || length(lambda) ~= num_features)
    lambda = zeros(size(Tx_all{1}, 2), 1);
end

accuracy = 0.01*norm(all_theta);
disp(['accuracy = ' num2str(accuracy)])

lambda = LDPpstar(hx, Tx_all, all_theta, accuracy, lambda);
disp(lambda)