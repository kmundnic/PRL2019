clear; close all; clc;
addpath('../../shadedErrorBar/')
addpath(genpath('../../matlab2tikz/src/'))

task = 'TaskA';
loss = 'CKL';
fraction = 5 * logspace(-4, -1, 10) * 100;
fraction = fraction(1:8);
mu = [2,10];
mu_idx = 2;
colors = get(gca,'colororder');

for mu_idx = 1:length(mu)
    figure(mu_idx);
    load(sprintf('~/Documents/Research/PRL2019/results/simulations_constant/%s/results_%s.mat', loss, task));
    shadedErrorBar(fraction, squeeze(mse(1,mu_idx,:,:))', {@mean, @std},...
        'lineprops', {'Color', colors(1,:)}); hold on; grid on;

    shadedErrorBar(fraction, squeeze(mse(2,mu_idx,:,:))', {@mean, @std},...
        'lineprops', {'Color', colors(2,:)}); hold on;

    shadedErrorBar(fraction, squeeze(mse(3,mu_idx,:,:))', {@mean, @std},...
        'lineprops', {'Color', colors(3,:)}); hold on;

    xlabel('Fraction of triplets');
    ylabel('MSE');
    title('Simulation results with constant model');

    matlab2tikz(sprintf('%s_%s_mu%d_simulations_constant.tex', loss, task, mu(mu_idx)))
    close(mu_idx);
end