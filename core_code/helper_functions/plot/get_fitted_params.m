models = [6];

for model = 1:length(models)
    m = models(model);
    for t = 1:nBins
        coef_names = Fitted_models(t, m).g.CoefficientNames;
        for i = 1:length(coef_names) % columns = coefficient
            fitted_betas(t, i) = Fitted_models(t, m).g.Coefficients.Estimate(i);
            fitted_SEs(t, i) = Fitted_models(t, m).g.Coefficients.SE(i);
            fitted_t_vals(t, i) = Fitted_models(t, m).g.Coefficients.tStat(i);
            fitted_p_vals(t, i) = Fitted_models(t, m).g.Coefficients.pValue(i);
        end
    end
end

%% plot beta timecourses from interaction model
coef_names;
figure;
hold on
yline(0, 'HandleVisibility', 'off')
for i = 1:length(coef_names)
    errorbar(1:50, fitted_betas(:, i), fitted_SEs(:, i), 'DisplayName', coef_names{i});
end

xlabel('time')
ylabel('beta value')
ylim([-0.5, 0.5])
xregion([1, 6]);
xregion([21,41], 'HandleVisibility', 'off');
xregion([14 20], 'FaceColor', 'r', 'FaceAlpha', 0.1)

legend({'intercept', ...
    'RT', 'coherence', 'performance', 'accuracy', 'volatility', ...
    'interaction', ... 
    'vol alone winning BIC',...
    'int winning by AIC&BIC'...
    }, 'Location', 'eastoutside')
