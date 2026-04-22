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

% test scatter plot
figure
scatter(x_use, y_use, 5, 'filled')
hold on
x_line = linspace(min(x_use), max(x_use), 200)';
y_line = beta_all(1) + beta_all(2) * x_line;

plot(x_line, y_line, 'r-', 'LineWidth', 2);

%% ===================== Z-score residual volatility =====================
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');

if sd_all == 0 || isnan(sd_all)
    resVol = zeros(size(resVol_mat));
else
    resVol = (resVol_mat - mu_all) ./ sd_all;
end


end
