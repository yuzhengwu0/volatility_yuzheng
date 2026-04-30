%% vol_distribution_scatter_plot
coh_weuse = cfg.coh_weuse;
cond = cfg.cond;
evidence_strength = cfg.evidence_strength;
volatility_strength = volatility_strength;
resVol = cfg.resVol;
x_use = cfg.x_use;
y_use = cfg.y_use;
beta_all = cfg.beta_all;


%% test scatter plot
figure
scatter(x_use, y_use, 5, 'filled')
hold on
x_line = linspace(min(x_use), max(x_use), 200)';
y_line = beta_all(1) + beta_all(2) * x_line;

xline(0, '--', 'Color', [0 0 0 0.3]);
plot(x_line, y_line, 'k--', 'LineWidth', 3);
title("volatility distribution")
xlabel("motion energy mean")
ylabel("motion energy STD")

%% regression plot for each coherence level
unique_cond = unique(cond(~isnan(cond)));

figure;
tiledlayout(1, 2);

for a = 1:length(unique_cond)

    this_cond = unique_cond(a);

    % trial condition
    idx_trial = cond == this_cond;

    % given coh trial × all bins
    x_cond = evidence_strength(idx_trial, :);
    y_cond = volatility_strength(idx_trial, :);

    % scatter
    x_plot = x_cond(:);
    y_plot = y_cond(:);

    good = ~isnan(x_plot) & ~isnan(y_plot);
    x_plot = x_plot(good);
    y_plot = y_plot(good);

    nexttile;
    scatter(x_plot, y_plot, 1, 'filled');
    hold on;

    
    % global
    y_line_all = beta_all(1) + beta_all(2) * x_line;
    plot(x_line, y_line_all, 'k--', 'LineWidth', 1);
    
    % panel-specific
    beta_this = polyfit(x_plot, y_plot, 1);
    y_line_this = beta_this(2) + beta_this(1) * x_line;
    plot(x_line, y_line_this, 'r-', 'LineWidth', 1.3);

    title(sprintf('vol = %.0f', this_cond));
    xlabel('motion energy mean');
    ylabel('motion energy STD');
    ylim([-2e-5 2e-4]);
    xlim([-2e-4 7e-4])
    xline(0, '--', 'Color', [0 0 0 0.3]);

    hold off;
end



%% coherence level and low/high vol

unique_coh  = unique(coh_weuse(~isnan(coh_weuse)));
unique_cond = unique(cond(~isnan(cond)));

figure;
tiledlayout(length(unique_coh), length(unique_cond));

for a = 1:length(unique_coh)

    this_coh = unique_coh(a);

    for c = 1:length(unique_cond)

        this_cond = unique_cond(c);

        idx_trial = (coh_weuse == this_coh) & (cond == this_cond);

        x_tmp = evidence_strength(idx_trial, :);
        y_tmp = volatility_strength(idx_trial, :);

        x_plot = x_tmp(:);
        y_plot = y_tmp(:);

        good = ~isnan(x_plot) & ~isnan(y_plot);
        x_plot = x_plot(good);
        y_plot = y_plot(good);

        nexttile;
        scatter(x_plot, y_plot, 1, 'filled');
        hold on;
        
        % global
        y_line_all = beta_all(1) + beta_all(2) * x_line;
        plot(x_line, y_line_all, 'k--', 'LineWidth', 1);
        
        % panel-specific
        beta_this = polyfit(x_plot, y_plot, 1);
        y_line_this = beta_this(2) + beta_this(1) * x_line;
        plot(x_line, y_line_this, 'r-', 'LineWidth', 1.3);

        xline(0, '--', 'Color', [0 0 0 0.3]);

        title(sprintf('coh = %.2f, vol = %.0f', this_coh, this_cond));
        xlabel('motion energy mean');
        ylabel('motion energy STD');

        ylim([-2e-5 2e-4]);
        xlim([-2e-4 7e-4]);

        hold off;
    end
end