clear; close all; clc;
addpath('../shadedErrorBar/')
addpath(genpath('../matlab2tikz/src/'))

fraction = 5 * logspace(-4, -1, 10);
colors = get(gca,'colororder');

load ../../results/simulations_logistic/STE/results_TaskA.mat
shadedErrorBar(fraction(1:8), mse', {@mean, @std},...
    'lineprops', {'Color', colors(1,:)}); hold on; grid on;

load ../../results/simulations_logistic/GNMDS/results_TaskA.mat
shadedErrorBar(fraction(1:8), mse', {@mean, @std},...
    'lineprops', {'Color', colors(2,:)}); hold on;

load ../../results/simulations_logistic/tSTE/results_TaskA.mat
shadedErrorBar(fraction(1:8), squeeze(mse(1,:,:))', {@mean, @std},...
    'lineprops', {'Color', colors(3,:)}); hold on;

load ../../results/simulations_logistic/tSTE/results_TaskA.mat
shadedErrorBar(fraction(1:8), squeeze(mse(2,:,:))', {@mean, @std},...
    'lineprops', {'Color', colors(4,:)}); hold on;

xlabel('Fraction of triplets');
ylabel('MSE');
title('Simulation results with logistic model')
legend('STE', 'tSTE \alpha=2', 'tSTE \alpha=10', 'GNMDS');

matlab2tikz('simulations_logistic.tex')