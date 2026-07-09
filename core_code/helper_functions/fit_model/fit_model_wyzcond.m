function [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = ...
    fit_model_wyzcond(cfg)

% make sure all vectors are columns
minN    = cfg.minN;
ConfY   = cfg.ConfY;
trial_coh = cfg.trial_coh;
vol     = cfg.vol;
rtX     = cfg.rtX;
cond    = cfg.cond;
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

    useCond = modelSpec(m).use1(1);   % cond flag
    useV    = modelSpec(m).use1(2);   % V flag
    useVxCond = modelSpec(m).use2;    % interaction flag

    labels = {'b0 (Intercept)', 'b_{rt}'};
    coefVarNames = ["(Intercept)", "R"];
    baseFormula = "ConfY ~ 1 + R";

    if useCond
        labels{end+1} = 'b_{cond}';
        coefVarNames(end+1) = "cond";
    end
    if useV
        labels{end+1} = 'b_{vol}';
        coefVarNames(end+1) = "V";
    end
    if useVxCond
        labels{end+1} = 'b_{volxcond}';
        coefVarNames(end+1) = "Vxcond";
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
               ~isnan(cond)     & ...
               ~isnan(subjID);
        if sum(mask) < minN
            continue;
        end

        y   = ConfY(mask);
        R   = rtX(mask);
        coh = trial_coh(mask);
        condK = cond(mask);
        V   = Vk(mask);
        sID = subjID(mask);

        Vxcond = V .* condK;

        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);
            T = table(y, R, condK, V, Vxcond, S2, S3, ...
                'VariableNames', {'ConfY','R','cond','V','Vxcond','S2','S3'});
        else
            T = table(y, R, condK, V, Vxcond, ...
                'VariableNames', {'ConfY','R','cond','V','Vxcond'});
        end

        f = baseFormula;
        if useCond
            f = f + " + cond";
        end
        if useV
            f = f + " + V";
        end
        if useVxCond
            f = f + " + Vxcond";
        end
        if useSubjDummies
            f = f + " + S2 + S3";
        end

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