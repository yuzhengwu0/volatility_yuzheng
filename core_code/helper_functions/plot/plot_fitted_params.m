models = [2];

for model = 1:length(models)
    m = models(model);
    t = nBins;
    coef_names = Fitted_models_incorrect(t, m).g.CoefficientNames;
    coef_names = string(coef_names);
    fitted_betas = NaN(t, length(coef_names));
    fitted_SEs = NaN(t, length(coef_names));
    fitted_t_vals = NaN(t, length(coef_names));
    fitted_p_vals = NaN(t, length(coef_names));
    for t = 1:nBins
       % if isempty(Fitted_models(t, m).g), continue; end  % skip unfitted bins
        for i = 1:length(coef_names)
            fitted_betas(t, i)  = Fitted_models_incorrect(t, m).g.Coefficients.Estimate(i);
            fitted_SEs(t, i)    = Fitted_models_incorrect(t, m).g.Coefficients.SE(i);
%            fitted_t_vals(t, i) = Fitted_models(t, m).g.Coefficients.tStat(i);
  %          fitted_p_vals(t, i) = Fitted_models(t, m).g.Coefficients.pValue(i);
        end
    end
end

% switch OUTCOME 
%     case 'acc'
%         fitted_betas = exp(fitted_betas);
% end

%% create universal mapping between variables & plot colors
canonicalNames  = ["(Intercept)", "C", "R", "coh", "z_cond", "V", 'Vxcoh', 'Vxz_cond', 'Vxcohxz_cond'];
canonicalLabels = ["Intercept", "accuracy", "RT", "coherence", "trial volatility", ...
                   "momentary vol", 'Vxcoh', 'Vxz_cond', 'Vxcohxz_cond'];
canonicalColors = lines(numel(canonicalNames));

%% plot beta timecourses from interaction model
figure; hold on
yline(0, 'HandleVisibility', 'off')

% add shaded areas to indicate winning model
%xregion([24 44], 'FaceAlpha', 0.1, 'DisplayName', 'CxVxCoh preferred by AIC')
%xregion([28 38], 'FaceAlpha', 0.1, 'DisplayName', 'CxVxCoh preferred by BIC & AIC', 'FaceColor', 'r')
xregion([29 37], 'DisplayName', 'Vxcoh model preffered by BIC', 'FaceAlpha', 0.1)
xregion([38 42], 'DisplayName', 'Vxz_cond model preffered by BIC', 'FaceAlpha', 0.1, 'FaceColor', 'b')


for i = 1:numel(coef_names)
    if i == 1, continue; end  % skip intercept

    % find where this coefficient lives in the canonical list
        idx = find(canonicalNames == coef_names(i), 1);   
        if isempty(idx), continue; end

    errorbar(1:nBins, fitted_betas(:,i), fitted_SEs(:,i), ...
        'DisplayName', canonicalLabels{idx}, ...
        'Color',       canonicalColors(idx,:), ...
        'LineWidth', 2);
end

legend('Location', 'eastoutside')

xlabel('time')
ylabel('beta value')
title(['model idx = ' num2str(m)])
%ylim([-0.5, 0.5])

%xregion([1 6]);
%xregion([21 41])
%xregion([14 20], 'FaceColor', 'r', 'FaceAlpha', 0.1)

% if m==5
%     legend({'RT', 'coherence', 'online performance', 'accuracy', 'momentary vol', ...
%         'trial vol',...
%         'PxV interaction',...
%         'vol alone winning BIC'}, 'Location', 'eastoutside');
% else
%         legend({'RT', 'coherence', 'accuracy', 'momentary vol', ...
%         'trial vol',...
%         'vol alone winning BIC',}, 'Location', 'eastoutside')
% end
