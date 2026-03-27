models = [5];

for model = 1:length(models)
    m = models(model);
    for t = 1:nBins
        coef_names = Fitted_models(t, m).g.CoefficientNames;
        for i = 1:length(coef_names) % columns = coefficient
            betas(t, i) = Fitted_models(t, m).g.Coefficients.Estimate(i);
            SEs(t, i) = Fitted_models(t, m).g.Coefficients.SE(i);
            t_vals(t, i) = Fitted_models(t, m).g.Coefficients.tStat(i);
            p_vals(t, i) = Fitted_models(t, m).g.Coefficients.pValue(i);
        end
    end
end

%% to plot error bars, do betas + SEs (upper bound) and betas - SEs (lower bound)