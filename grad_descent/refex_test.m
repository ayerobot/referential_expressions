Tx_all = {};
all_theta = zeros(3, 1);
for i = 1:12
    load(['~/cs/h2r/referential_expressions/grad_descent/test' num2str(i) '.mat'])
    Tx_all{i} = Tx;
    all_theta = all_theta + theta';
end
all_theta = all_theta/12;
hx = ones(length(Tx_all{1}), 1)/length(Tx_all{1});

lambda = LDPpstar(hx, Tx_all, all_theta, 0.001, lambda);
disp(lambda)