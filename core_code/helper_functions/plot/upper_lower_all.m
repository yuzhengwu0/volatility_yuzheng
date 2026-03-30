nPlot = numel(models_to_plot);

% compute winning bins
[~, bestAIC_idx] = min(AIC_mat, [], 2, 'omitnan');  % nBins x 1 - minimal AIC
[~, bestBIC_idx] = min(BIC_mat, [], 2, 'omitnan');  % nBins x 1 - minimal BIC

figure;
for p = 1:nPlot
    m = models_to_plot(p);

    fitted_betas  = [];
    fitted_SEs    = [];
    fitted_t_vals = [];
    fitted_p_vals = [];

    for t = 1:nBins
        coef_names = Fitted_models(t, m).g.CoefficientNames;
        for i = 1:length(coef_names)
            fitted_betas(t, i)  = Fitted_models(t, m).g.Coefficients.Estimate(i);
            fitted_SEs(t, i)    = Fitted_models(t, m).g.Coefficients.SE(i);
            fitted_t_vals(t, i) = Fitted_models(t, m).g.Coefficients.tStat(i);
            fitted_p_vals(t, i) = Fitted_models(t, m).g.Coefficients.pValue(i);
        end
    end

    subplot(2,1,p);
    hold on;
    yline(0, 'HandleVisibility', 'off');

    % current model m winning bins ──────────────────────────
    aic_win_bins = find(bestAIC_idx == m);  
    bic_win_bins = find(bestBIC_idx == m);  

    % black shadow：AIC winning bins
    for k = 1:length(aic_win_bins)
        xregion(aic_win_bins(k)-0.5, aic_win_bins(k)+0.5, ...
            'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.2, ...
            'HandleVisibility', 'off');
    end

    % red shadow：BIC winning bins
    for k = 1:length(bic_win_bins)
        xregion(bic_win_bins(k)-0.5, bic_win_bins(k)+0.5, ...
            'FaceColor', 'r', 'FaceAlpha', 0.15, ...
            'HandleVisibility', 'off');
    end
    % ─────────────────────────────────────────────────────────

    for i = 1:length(coef_names)
        errorbar(1:nBins, fitted_betas(:, i), fitted_SEs(:, i), ...
            'DisplayName', coef_names{i});
    end

    xlabel('time');
    ylabel('beta value');
    ylim([-0.5, 0.5]);
    xlim([1, nBins]);
    termStr = strjoin(coef_names, ', ');
    title(sprintf('Model %d:  %s', m, termStr), 'Interpreter', 'none');
    legend(coef_names, 'Location', 'eastoutside');
end