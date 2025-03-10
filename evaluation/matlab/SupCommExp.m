close all
clear all
clc

% Parameters
kappa = 20;
W = 250 * 10^3;
S0 = 50;
K = [1, 5, 10, 15, 20, 26];
eta = 0.01;

% Latency
N1 = 1;
delay1 = K * 2 * S0 ./ (W - N1 * kappa * K * 2 * S0);
N2 = 3;
delay2 = K * 2 * S0 ./ (W - N2 * kappa * K * 2 * S0);
delay2(delay2 < 0) = inf;
N3 = 6;
delay3 = K * 2 * S0 ./ (W - N3 * kappa * K * 2 * S0);
delay3(delay3 < 0) = inf;


% Packet error probability
P_fail = 1 - (1 - eta).^(K * 2);

% Plot
figure('Position', [680, 558, 800, 400]);
yyaxis left
plot(K, (delay1), '-.o', 'LineWidth', 2, 'MarkerSize',  10, 'DisplayName', ['Latency, N = ' num2str(N1)]);
hold on;
plot(K, (delay2), '-o', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', ['Latency, N = ' num2str(N2)]);
hold on;
plot(K, (delay3), '--o', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', ['Latency, N = ' num2str(N3)]);
ylabel('Latency [s]', 'FontSize',14);
ylim([-0.01 0.2]);
grid on;

yyaxis right
plot(K, P_fail, '-x', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'P_{fail}');
ylabel('P_{fail}', 'FontSize',14);
ylim([-0.05 0.45]);

xlim([0 27]);
xlabel('K', 'FontSize',14);
lgd = legend('show', 'Location', 'best');

lgd.FontSize = 14;
set(gca, 'FontSize',14, 'linewidth',1.1, 'Gridalpha',0.25);