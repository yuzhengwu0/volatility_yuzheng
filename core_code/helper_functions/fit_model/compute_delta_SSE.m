
delta_sse = NaN(nBins, nModels-1);
fStat = NaN(nBins, nModels-1);
pval = NaN(nBins, nModels-1);

np_full = Fitted_models(1, nModels).g.NumEstimatedCoefficients;

for m = 1:nModels-1
    np_nest = Fitted_models(1, m).g.NumEstimatedCoefficients;
    delta_p = np_nest - np_full;
    for i = 1:height(Fitted_models)
        % compute delta_sse
        sse_full = Fitted_models(i, nModels).g.SSE;
        sse_nest = Fitted_models(i, m).g.SSE;
        delta_sse(i, m) = sse_nest - sse_full;

        % compute fStat
        mse_full = Fitted_models(i, nModels).g.MSE;
        fStat(i, m) = (delta_sse(i, m) / delta_p) / mse_full;

        % compute p-value for fStat
        pval(i, m) = fpdf(i, np_full, np_nest);

    end
end

%% correct p values and plot F-stat & p
correction = 'bonferroni';
switch correction
    case 'bonferroni'
        alpha = .05 / 50;
end

% plot F-stat
figure;
hold on
for i = 1:3
    plot(1:50, fStat(:, i), 'LineWidth', 2);
end
legend({'M0 vs. M3', 'M1 vs. M3', 'M2 vs. M3'})

% plot corrected p-vals
corrected_p = pval < alpha;
figure;
hold on
for i = 1:3
    plot(1:50, corrected_p(:, i), 'LineWidth', 2);
end
legend({'M0 vs. M3', 'M1 vs. M3', 'M2 vs. M3'})




