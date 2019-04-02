clear; close all; clc;
addpath('../../shadedErrorBar/')
addpath(genpath('../../matlab2tikz/src/'))

task = 'TaskA';
loss = 'tSTE';
fraction = 5 * logspace(-4, -1, 10)*100;
fraction = fraction(1:8);
alpha = [2,10];
alpha_idx = 1;
colors = get(gca,'colororder');

for alpha_idx = 1:length(alpha)
    figure(alpha_idx);
    load(sprintf('~/Documents/Research/PRL2019/results/simulations_constant/%s/results_%s.mat', loss, task));
    shadedErrorBar(fraction, squeeze(mse(1,alpha_idx,:,:))', {@mean, @std},...
        'lineprops', {'Color', colors(1,:)}); hold on; grid on;

    shadedErrorBar(fraction, squeeze(mse(2,alpha_idx,:,:))', {@mean, @std},...
        'lineprops', {'Color', colors(2,:)}); hold on;

    shadedErrorBar(fraction, squeeze(mse(3,alpha_idx,:,:))', {@mean, @std},...
        'lineprops', {'Color', colors(3,:)}); hold on;

    xlabel('Fraction of triplets');
    ylabel('MSE');
    title('Simulation results with constant model');

    matlab2tikz(sprintf('%s_%s_a%d_simulations_constant.tex', loss, task, alpha(alpha_idx)))
    close(alpha_idx);
end