% Tx_all => size 12 cell array. Each cell contains the value of the
%           features on the discretization of the board for a given command.
% all_theta => average value of features at every data point (over all commands).
% hx => uniform distribution over all data points
% 

Tx_all = {};
all_theta = zeros(3, 1);
for i = 1:12
    load(['command' num2str(i) '.mat'])
    Tx_all{i} = Tx;
    all_theta = all_theta + theta';
end
all_theta = all_theta/12;
hx = ones(length(Tx_all{1}), 1)/length(Tx_all{1});

% if you preset a lambda, it won't be overwritten. This is helpful b/c you
% can initialize lambda to some value so you don't always have to start at
% the beginning.
if ~exist('lambda', 'var'); lambda = zeros(size(Tx_all{i}, 2), 1); end;

lambda = LDPpstar(hx, Tx_all, all_theta, 0.001, lambda);
disp(lambda)