models = [5];

for model = 1:length(models)
    m = models(model);
    t = nBins;
    coef_names = Fitted_models(t, m).g.CoefficientNames;
    coef_names = string(coef_names);
    fitted_betas = NaN(t, length(coef_names));
    fitted_SEs = NaN(t, length(coef_names));
    fitted_t_vals = NaN(t, length(coef_names));
    fitted_p_vals = NaN(t, length(coef_names));
    for t = 1:nBins
       % if isempty(Fitted_models(t, m).g), continue; end  % skip unfitted bins
        for i = 1:length(coef_names)
            fitted_betas(t, i)  = Fitted_models(t, m).g.Coefficients.Estimate(i);
            fitted_SEs(t, i)    = Fitted_models(t, m).g.Coefficients.SE(i);
            fitted_t_vals(t, i) = Fitted_models(t, m).g.Coefficients.tStat(i);
            fitted_p_vals(t, i) = Fitted_models(t, m).g.Coefficients.pValue(i);
        end
    end
end

%% create universal mapping between variables & plot colors
canonicalNames  = ["(Intercept)", "R", "coh", "z_cond", "V", "Vxcoh","Vxz_cond","Vxcohxz_cond"];
canonicalLabels = ["Intercept", "RT", "coh", "cond", ...
                   "vol","volxcoh","volxcond","volxcohxcond"];

canonicalColors = lines(numel(canonicalNames));

%% plot beta timecourses from interaction model
figure; hold on
yline(0, 'HandleVisibility', 'off')

% add shaded areas to indicate winning model
xregion([29 37], 'FaceAlpha', 0.2, 'DisplayName', 'volxcoh preferred by BIC','FaceColor', [0.6 0.8 1.0])
xregion([38 42], 'FaceAlpha', 0.2, 'DisplayName', 'volxcond preferred by BIC', 'FaceColor', [1.0 0.6 0.6])

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
title('beta plot of blend model M4 - volxcohxcond')
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
