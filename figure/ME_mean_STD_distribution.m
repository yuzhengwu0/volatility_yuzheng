x = evidence_strength(:);
y = volatility_strength(:);

mask = ~isnan(x) & ~isnan(y);

scatter(x(mask), y(mask), 5, 'k', 'filled')
xlabel('evidence')
ylabel('volatility')
title('volatility vs evidence')

% linear reg
hold on

p = polyfit(x(mask), y(mask), 1);   % 1 = linear
xline = linspace(min(x), max(x), 100);
yline = polyval(p, xline);

plot(xline, yline, 'r-', 'LineWidth', 2)

% polynomial fit
p2 = polyfit(x(mask), y(mask), 2);
yline2 = polyval(p2, xline);

plot(xline, yline2, 'b-', 'LineWidth', 2)

legend({'data','linear','quadratic'})