models = [4];
for model = 1:length(models)
    m = models(model);
    nBins = size(Fitted_models, 1);   % <-- derive directly, don't rely on a workspace variable

    % find a bin that actually has a fitted model, to get coef_names
    validBin = find(~cellfun(@isempty, {Fitted_models(:, m).g}), 1, 'last');
    if isempty(validBin)
        error('No fitted models found for model index %d', m);
    end
    coef_names = Fitted_models(validBin, m).g.CoefficientNames;
    coef_names = string(coef_names);

    fitted_betas  = NaN(nBins, length(coef_names));
    fitted_SEs    = NaN(nBins, length(coef_names));
    fitted_t_vals = NaN(nBins, length(coef_names));
    fitted_p_vals = NaN(nBins, length(coef_names));

    for t = 1:nBins
        if isempty(Fitted_models(t, m).g)
            continue;  % skip unfitted bins
        end
        theseCoefNames = string(Fitted_models(t, m).g.CoefficientNames);
        for i = 1:length(coef_names)
            % match by name, not position — bins can have different #s of terms if a fit failed
            idxInThisBin = find(theseCoefNames == coef_names(i), 1);
            if isempty(idxInThisBin), continue; end
            fitted_betas(t, i)  = Fitted_models(t, m).g.Coefficients.Estimate(idxInThisBin);
            fitted_SEs(t, i)    = Fitted_models(t, m).g.Coefficients.SE(idxInThisBin);
            fitted_t_vals(t, i) = Fitted_models(t, m).g.Coefficients.tStat(idxInThisBin);
            fitted_p_vals(t, i) = Fitted_models(t, m).g.Coefficients.pValue(idxInThisBin);
        end
    end
end

%% create universal mapping between variables & plot colors
canonicalNames  = ["(Intercept)", "R", "cond", "V", "Vxcond"];
canonicalLabels = ["Intercept", "rt", "cond", "Vt", "Vt x cond"];
canonicalColors = lines(numel(canonicalNames));

%% plot beta timecourses from interaction model
figure; hold on
yline(0, 'HandleVisibility', 'off')

% add shaded areas to indicate winning model
% xregion([30 43], 'FaceAlpha', 0.1, 'DisplayName', 'preferred by AIC')

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
sgtitle('beta value plot')
% ylim([-0.4, 0.4])

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
