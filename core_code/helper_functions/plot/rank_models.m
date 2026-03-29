function [deltaTbl, score, rankIdx, top4Idx] = rank_models(AIC_mat, BIC_mat, modelNames)

minAIC_perBin = min(AIC_mat, [], 2, 'omitnan');
minBIC_perBin = min(BIC_mat, [], 2, 'omitnan');

deltaAIC_mat = AIC_mat - minAIC_perBin;
deltaBIC_mat = BIC_mat - minBIC_perBin;

meanDeltaAIC = mean(deltaAIC_mat, 1, 'omitnan');
medDeltaAIC  = median(deltaAIC_mat, 1, 'omitnan');
meanDeltaBIC = mean(deltaBIC_mat, 1, 'omitnan');
medDeltaBIC  = median(deltaBIC_mat, 1, 'omitnan');

deltaTbl = table(modelNames(:), meanDeltaAIC(:), medDeltaAIC(:), ...
    meanDeltaBIC(:), medDeltaBIC(:), ...
    'VariableNames', {'Model','Mean_delta_AIC','Median_delta_AIC', ...
    'Mean_delta_BIC','Median_delta_BIC'});

disp('=== Delta AIC/BIC summary ===');
disp(deltaTbl);

score = mean([ ...
    meanDeltaAIC(:), ...
    medDeltaAIC(:), ...
    meanDeltaBIC(:), ...
    medDeltaBIC(:) ...
    ], 2, 'omitnan');

[~, rankIdx] = sort(score, 'ascend', 'MissingPlacement','last');
rankIdx = rankIdx(~isnan(score(rankIdx)));

N_TOP = 4;
top4Idx = rankIdx(1:min(N_TOP, numel(rankIdx)));

fprintf('\n=== Top 4 models ===\n');
disp(table(modelNames(top4Idx)', score(top4Idx), ...
    'VariableNames', {'Model','CompositeScore'}));

end