function [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = ...
    fit_model_blend_corr(cfg)

% make sure all vectors are columns
minN    = cfg.minN;
ConfY   = cfg.ConfY;
Correct = cfg.Correct;
z_coh   = cfg.coh;
z_perf  = cfg.z_perf;
rtX     = cfg.rtX;
subjID  = cfg.subjID;
resVol  = cfg.resVol;
cond    = cfg.cond;
modelNames = cfg.modelNames;
modelSpec = cfg.modelSpec;
useSubjDummies = cfg.useSubjDummies;

if nargin < 9 || isempty(minN)
    minN = 50;
end

[~, K] = size(resVol);
nModels = numel(modelNames);

AIC_mat  = nan(K, nModels);
BIC_mat  = nan(K, nModels);
Nobs_mat = nan(K, nModels);

Models = struct();
Fitted_models = struct();

for m = 1:nModels
    fprintf('\n=== Fitting %s ===\n', modelNames{m});


    labels = {'b0 (Intercept)', 'b_{rt}', 'b_{coh}', 'b_{cond}'};
    coefVarNames = ["(Intercept)", "R", "coh", "cond"];
    baseFormula = "ConfY ~ 1 + R + coh + cond";

    % optional one-way terms

    if modelSpec(m).use1(1)
        labels{end+1} = 'b_{vol}';
        coefVarNames(end+1) = "V";
    end


    % add Vxcoh
    if modelSpec(m).use2(1)
        labels{end+1} = 'b_{volxcoh}';
        coefVarNames(end+1) = "Vxcoh";
    end

    % add Vxz_con
    if modelSpec(m).use2(2)
        labels{end+1} = 'b_{volxcond}';
        coefVarNames(end+1) = "Vxcond";
    end


    % add Vxcohxcond
    if modelSpec(m).use3
        labels{end+1} = 'b_{volxcohxcond}';
        coefVarNames(end+1) = "Vxcohxcond";
    end


    nTerms   = numel(labels);
    betas    = nan(K, nTerms);
    beta_ses = nan(K, nTerms);

    for k = 1:K
        Vk = resVol(:, k);

        mask = ~isnan(Vk)      & ...
               ~isnan(ConfY)   & ...
               ~isnan(z_coh)   & ...
               ~isnan(z_perf)  & ...
               ~isnan(rtX)     & ...
               ~isnan(subjID);

        if sum(mask) < minN
            continue;
        end

        % current-bin data
        y   = ConfY(mask);
        R   = rtX(mask);
        coh = z_coh(mask);
        V   = Vk(mask);
        sID = subjID(mask);
        cond = cond(mask);

        % interactions
        Vxcond     = V.* cond;
        Vxcoh     = V .* coh;
        Vxcohxcond = V.*coh.*cond;


        % table
        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);

            T = table(y, R, coh, cond, V, Vxcoh, Vxcond,Vxcohxcond,S2, S3, ...
                'VariableNames', {'ConfY','R','coh','cond','V', ...
                                  'Vxcoh', 'Vxcond', 'Vxcohxcond', 'S2','S3'});
        else
            T = table(y, R, coh, cond, V, Vxcoh, Vxcond, Vxcohxcond, ...
                'VariableNames', {'ConfY','R','coh','cond','V', ...
                                   'Vxcoh', 'Vxcond', 'Vxcohxcond',});
        end

        % -------------------------------------------------
        % Build formula
        % -------------------------------------------------
        f = baseFormula;

       if modelSpec(m).use1(1)
            f = f + " + V";
        end

        if modelSpec(m).use2(1)
            f = f + " + Vxcoh";
        end

        if modelSpec(m).use2(2)
            f = f + " + Vxcond";
        end

       if sum(modelSpec(m).use3) == 1
            f = f + " + Vxcohxcond";
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