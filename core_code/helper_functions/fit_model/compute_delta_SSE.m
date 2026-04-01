delta_sse = NaN(50, 3);
fStat = NaN(50, 3);
pval = NaN(50,3);

np_full = Fitted_models(1, 4).g.NumEstimatedCoefficients;

for m = 1:nModels-1
    np_nest = Fitted_models(1, m).g.NumEstimatedCoefficients;
    delta_p = np_nest - np_full;
    for i = 1:height(Fitted_models)
        % compute delta_sse
        sse_full = Fitted_models(i, 4).g.SSE;
        sse_nest = Fitted_models(i, m).g.SSE;
        delta_sse(i, m) = sse_nest - sse_full;

        % compute fStat
        mse_full = Fitted_models(i, 4).g.MSE;
        fStat(i, m) = (delta_sse(i, m) / delta_p) / mse_full;

        % compute p-value for fStat
        pval(i, m) = fpdf(i, np_full, np_nest);

    end
end

%% compute F-statistic

for m = 1:nModels-1
    for i = 1:height(Fitted_models)
    end
end