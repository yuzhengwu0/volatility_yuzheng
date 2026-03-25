function SelOut = run_split_and_plot(idxSplit, splitTag, ...
    ConfY, Correct, subjID, split_perf, Cz_all, RTz_all, resVol_time, t_norm, colSub, ...
    modelNames, modelSpec, baseLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
    useSubjDummies, minN_pooled, minN_sub, ...
    FORCE_FIXED_MODELS, fixedTopIdx, DO_PLOT_AICBIC_DOTS, cfg)

% unpack config contents to be used in the function
%ConfY = cfg.ConfY;
%Correct = cfg.Correct;



% ---- pooled model selection ----
[AIC_mat, BIC_mat, Pool] = fit_models_pooled( ...
    idxSplit, cfg);

K = size(AIC_mat, 1);
nModels = numel(modelNames);

if DO_PLOT_AICBIC_DOTS

    % For each time bin, find which model has the minimum AIC / BIC
    [~, bestAIC_idx] = min(AIC_mat, [], 2, 'omitnan');   % K x 1
    [~, bestBIC_idx] = min(BIC_mat, [], 2, 'omitnan');   % K x 1

    % Fix rows that are all NaN
    allNanAIC = all(isnan(AIC_mat), 2);
    allNanBIC = all(isnan(BIC_mat), 2);

    bestAIC_idx(allNanAIC) = NaN;
    bestBIC_idx(allNanBIC) = NaN;

    figAB = figure('Color','w','Position',[180 160 1000 420]); hold on;

    % y-axis: M0 at bottom, M9 at top
    yModel = 1:nModels;

    % dummy handles for legend
    hAIC = plot(nan, nan, 'o', ...
        'MarkerSize', 5, ...
        'MarkerFaceColor', [0 0.4470 0.7410], ...
        'MarkerEdgeColor', 'none', ...
        'DisplayName', 'Min AIC');

    hBIC = plot(nan, nan, 'o', ...
        'MarkerSize', 5, ...
        'MarkerFaceColor', [0.8500 0.3250 0.0980], ...
        'MarkerEdgeColor', 'none', ...
        'DisplayName', 'Min BIC');

    % plot blue dots for min AIC
    for k = 1:K
        if ~isnan(bestAIC_idx(k))
            plot(k, bestAIC_idx(k)-0.10, 'o', ...
                'MarkerSize', 5, ...
                'MarkerFaceColor', [0 0.4470 0.7410], ...
                'MarkerEdgeColor', 'none', ...
                'HandleVisibility', 'off');
        end
    end

    % plot red dots for min BIC
    for k = 1:K
        if ~isnan(bestBIC_idx(k))
            plot(k, bestBIC_idx(k)+0.10, 'o', ...
                'MarkerSize', 5, ...
                'MarkerFaceColor', [0.8500 0.3250 0.0980], ...
                'MarkerEdgeColor', 'none', ...
                'HandleVisibility', 'off');
        end
    end

    set(gca, ...
        'YTick', yModel, ...
        'YTickLabel', modelNames, ...
        'YLim', [0.5 nModels+0.5], ...
        'XTick', 1:K, ...
        'XLim', [0.5 K+0.5], ...
        'FontSize', 11, ...
        'LineWidth', 1, ...
        'TickLabelInterpreter','none');

    xlabel('Time bin');
    ylabel('Model');
    title(sprintf('%s: Best model per time bin | AIC (blue) and BIC (red)', splitTag), ...
        'Interpreter','none');

    grid on;
    box off;

    legend([hAIC, hBIC], {'Min AIC','Min BIC'}, 'Location','eastoutside');

    outPDF_ab = sprintf('AIC_BIC_bestModel_dots_%s.pdf', splitTag);
    set(figAB,'Renderer','painters');
    print(figAB, outPDF_ab, '-dpdf', '-painters');
    fprintf('✓ Saved AIC/BIC dot plot: %s\n', outPDF_ab);
end

% ---- delta summary ----
minAIC_perBin = min(AIC_mat, [], 2, 'omitnan');
minBIC_perBin = min(BIC_mat, [], 2, 'omitnan');
deltaAIC_mat  = AIC_mat - minAIC_perBin;
deltaBIC_mat  = BIC_mat - minBIC_perBin;

meanDeltaAIC = mean(deltaAIC_mat, 1, 'omitnan');
medDeltaAIC  = median(deltaAIC_mat, 1, 'omitnan');
meanDeltaBIC = mean(deltaBIC_mat, 1, 'omitnan');
medDeltaBIC  = median(deltaBIC_mat, 1, 'omitnan');

deltaTbl = table(modelNames(:), meanDeltaAIC(:), medDeltaAIC(:), meanDeltaBIC(:), medDeltaBIC(:), ...
    'VariableNames', {'Model','Mean_dAIC','Median_dAIC','Mean_dBIC','Median_dBIC'});

disp(['=== ' splitTag ' Delta AIC/BIC summary ===']);
disp(deltaTbl);

% ---- choose top models ----
if FORCE_FIXED_MODELS
    topIdx = fixedTopIdx(:);
else
    score = mean([meanDeltaAIC(:), medDeltaAIC(:), meanDeltaBIC(:), medDeltaBIC(:)], 2, 'omitnan');
    [~, rankIdx] = sort(score, 'ascend', 'MissingPlacement','last');
    rankIdx = rankIdx(~isnan(score(rankIdx)));
    N_TOP = 2;
    topIdx = rankIdx(1:min(N_TOP, numel(rankIdx)));
end

% ---- per-subject per-bin refit ----
Sel = refit_models_perSubjectPerBin( ...
    idxSplit, ConfY, Correct, subjID, split_perf, Cz_all, RTz_all, resVol_time, t_norm, ...
    modelNames, modelSpec, baseLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
    topIdx, minN_sub);

SelOut = Sel;
end