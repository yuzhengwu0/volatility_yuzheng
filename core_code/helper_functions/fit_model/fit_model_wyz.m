function [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = ...
    fit_model_wyz(cfg)

% make sure all vectors are columns
minN    = cfg.minN;
ConfY   = cfg.ConfY;
trial_coh = cfg.trial_coh;
vol     = cfg.vol;
rtX     = cfg.rtX;
subjID  = cfg.subjID;
modelNames = cfg.modelNames;
modelSpec = cfg.modelSpec;
useSubjDummies = cfg.useSubjDummies;

minN = cfg.minN;
if isempty(minN)
    minN = 50;
end

[~, K] = size(vol);
nModels = numel(modelNames);

AIC_mat  = nan(K, nModels);
BIC_mat  = nan(K, nModels);
Nobs_mat = nan(K, nModels);

Models = struct();
Fitted_models = struct();

for m = 1:nModels
    fprintf('\n=== Fitting %s ===\n', modelNames{m});


    labels = {'b0 (Intercept)', 'b_{rt}', 'b_{coh}'};
    coefVarNames = ["(Intercept)", "R", "coh"];
    baseFormula = "ConfY ~ 1 + R + coh";

    % optional one-way terms

    if modelSpec(m).use1
        labels{end+1} = 'b_{vol}';
        coefVarNames(end+1) = "V";
    end


    % add Vxcoh
    if modelSpec(m).use2
        labels{end+1} = 'b_{volxcoh}';
        coefVarNames(end+1) = "Vxcoh";
    end



    nTerms   = numel(labels);
    betas    = nan(K, nTerms);
    beta_ses = nan(K, nTerms);

    for k = 1:K
        Vk = vol(:, k);

        mask = ~isnan(Vk)      & ...
               ~isnan(ConfY)   & ...
               ~isnan(trial_coh)   & ...
               ~isnan(rtX)     & ...
               ~isnan(subjID);

        if sum(mask) < minN
            continue;
        end

        % current-bin data
        y   = ConfY(mask);
        R   = rtX(mask);
        coh = trial_coh(mask);
        V   = Vk(mask);
        sID = subjID(mask);

        % interactions
        Vxcoh     = V .* coh;


        % table
        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);

            T = table(y, R, coh, V, Vxcoh, S2, S3, ...
                'VariableNames', {'ConfY','R','coh','V', ...
                                  'Vxcoh', 'S2','S3'});
        else
            T = table(y, R, coh, V, Vxcoh, ...
                'VariableNames', {'ConfY','R','coh','V', ...
                                   'Vxcoh',});
        end

        % -------------------------------------------------
        % Build formula
        % -------------------------------------------------
        f = baseFormula;

       if modelSpec(m).use1
            f = f + " + V";
        end

        if modelSpec(m).use2
            f = f + " + Vxcoh";
        end

        if useSubjDummies
            f = f + " + S2 + S3";
        end

        % fit
        try
            g = fitlm(T, f);
        catch ME
            fprintf('fitglm failed | model=%s | bin=%d\n', modelNames{m}, k);
            fprintf('%s\n', ME.message);
            continue;
        end

        Fitted_models(k, m).g = g;

        AIC_mat(k, m)  = g.ModelCriterion.AIC;
        BIC_mat(k, m)  = g.ModelCriterion.BIC;
        Nobs_mat(k, m) = sum(mask);

end

end