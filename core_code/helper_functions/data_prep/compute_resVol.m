function [resVol_mat, resVol, evidence_strength, volatility_strength] = compute_resVol(cfg)
% Compute residual volatility across all trials
% Also recode volatility into cond:
%   low vol  -> cond = 1
%   high vol -> cond = 2
%
% Outputs:
%   resVol_mat  : raw residual volatility (trial x time bin)
%   resVol_time : z-scored residual volatility across all trials and bins
%   cond        : recoded volatility condition

evidence_strength = cfg.evidence_strength;
volatility_strength = cfg.volatility_strength;
coh_weuse = cfg.coh_weuse;
cond = cfg.cond;


%% ===================== Residual volatility =====================
x_all = evidence_strength(:);
% x_all = abs(evidence_strength(:));
y_all = volatility_strength(:);

mask_all = ~isnan(x_all) & ~isnan(y_all);

x_use = x_all(mask_all);
y_use = y_all(mask_all);

Xall = [ones(sum(mask_all),1), x_use];
beta_all = Xall \ y_use;

yhat_all = Xall * beta_all;
resid_all = y_use - yhat_all;

tmp_all = nan(size(x_all));
tmp_all(mask_all) = resid_all;

resVol_mat = reshape(tmp_all, size(evidence_strength));


%% test scatter plot
figure
scatter(x_use, y_use, 5, 'filled')
hold on
x_line = linspace(min(x_use), max(x_use), 200)';
y_line = beta_all(1) + beta_all(2) * x_line;

xline(0, '--', 'Color', [0 0 0 0.3]);
plot(x_line, y_line, 'k--', 'LineWidth', 3);


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

    x_line = linspace(min(x_plot), max(x_plot), 200)';
    y_line = beta_all(1) + beta_all(2) * x_line;

    plot(x_line, y_line, 'r-', 'LineWidth', 2);

    title(sprintf('vol = %.0f', this_cond));
    xlabel('motion energy mean');
    ylabel('motion energy STD');
    ylim([-2e-5 2e-4]);
    xlim([-2e-4 7e-4])
    xline(0, '--', 'Color', [0 0 0 0.3]);

    hold off;
end



%% split by low vol and high vol

% regression plot for each coherence level
unique_coh = unique(coh_weuse(~isnan(coh_weuse)));

figure;
tiledlayout(2, 3);

for a = 1:length(unique_coh)

    this_coh = unique_coh(a);

    % by trial
    idx_trial = coh_weuse == this_coh;

    % all trial × all bins
    x_coh = evidence_strength(idx_trial, :);
    y_coh = volatility_strength(idx_trial, :);

    % scatter
    x_plot = x_coh(:);
    y_plot = y_coh(:);

    good = ~isnan(x_plot) & ~isnan(y_plot);
    x_plot = x_plot(good);
    y_plot = y_plot(good);

    nexttile;
    scatter(x_plot, y_plot, 1, 'filled');
    hold on;

    x_line = linspace(min(x_plot), max(x_plot), 200)';
    y_line = beta_all(1) + beta_all(2) * x_line;

    plot(x_line, y_line, 'r-', 'LineWidth', 2);

    title(sprintf('Coherence = %.2f', this_coh));
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
%% ===================== Z-score residual volatility =====================
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');

if sd_all == 0 || isnan(sd_all)
    resVol = zeros(size(resVol_mat));
else
    resVol = (resVol_mat - mu_all) ./ sd_all;
end


end
