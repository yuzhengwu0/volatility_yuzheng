function [Models, Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = ...
    fit_model_corr(cfg)

% make sure all vectors are columns
minN    = cfg.minN;
ConfY   = cfg.ConfY;
Correct = cfg.Correct;
z_coh   = cfg.z_coh;
z_perf  = cfg.z_perf;
rtX     = cfg.rtX;
subjID  = cfg.subjID;
resVol  = cfg.resVol;
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

    % -------------------------------------------------
    % Decide base terms for this model
    % M0-M4: 1 + C + R + coh
    % M5(use3): 1 + C + R
    % -------------------------------------------------
    if modelSpec(m).use3
        labels = {'b0 (Intercept)', 'b_{corr}', 'b_{rt}'};
        coefVarNames = ["(Intercept)", "C", "R"];
        baseFormula = "ConfY ~ 1 + C + R";
    else
        labels = {'b0 (Intercept)', 'b_{corr}', 'b_{rt}', 'b_{coh}'};
        coefVarNames = ["(Intercept)", "C", "R", "coh"];
        baseFormula = "ConfY ~ 1 + C + R + coh";
    end

    % optional one-way terms
    if modelSpec(m).use1(1)
        labels{end+1} = 'b_{perf}';
        coefVarNames(end+1) = "P";
    end

    if modelSpec(m).use1(2)
        labels{end+1} = 'b_{vol}';
        coefVarNames(end+1) = "V";
    end

    % optional two-way term
    if modelSpec(m).use2(1)
        labels{end+1} = 'b_{perf×vol}';
        coefVarNames(end+1) = "PxV";
    end

    if modelSpec(m).use2(2)
        labels{end+1} = 'b_{corr×vol}';
        coefVarNames(end+1) = "CxV";
    end

    % optional three-way term
    if modelSpec(m).use3
        labels{end+1} = 'b_{perf×vol×coh}';
        coefVarNames(end+1) = "PxVxcoh";
    end

    nTerms   = numel(labels);
    betas    = nan(K, nTerms);
    beta_ses = nan(K, nTerms);

    for k = 1:K
        Vk = resVol(:, k);

        mask = ~isnan(Vk)      & ...
               ~isnan(ConfY)   & ...
               ~isnan(Correct) & ...
               ~isnan(z_coh)   & ...
               ~isnan(z_perf)  & ...
               ~isnan(rtX)     & ...
               ~isnan(subjID);

        if sum(mask) < minN
            continue;
        end

        % current-bin data
        y   = ConfY(mask);
        C   = Correct(mask);
        R   = rtX(mask);
        coh = z_coh(mask);
        P   = z_perf(mask);
        V   = Vk(mask);
        sID = subjID(mask);

        % interactions
        PxV     = P .* V;
        CxV     = C .* V;
        PxVxcoh = P .* V .* coh;

        % table
        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);

            T = table(y, C, R, coh, P, V, PxV, CxV, PxVxcoh, S2, S3, ...
                'VariableNames', {'ConfY','C','R','coh','P','V', ...
                                  'PxV','CxV','PxVxcoh','S2','S3'});
        else
            T = table(y, C, R, coh, P, V, PxV, CxV, PxVxcoh, ...
                'VariableNames', {'ConfY','C','R','coh','P','V', ...
                                  'PxV','CxV','PxVxcoh'});
        end

        % -------------------------------------------------
        % Build formula
        % -------------------------------------------------
        f = baseFormula;

        if modelSpec(m).use1(1)
            f = f + " + P";
        end

        if modelSpec(m).use1(2)
            f = f + " + V";
        end

        if modelSpec(m).use2(1)
            f = f + " + PxV";
        end

        if modelSpec(m).use2(2)
            f = f + " + CxV";
        end

        if modelSpec(m).use3
            f = f + " + PxVxcoh";
        end

        if useSubjDummies
            f = f + " + S2 + S3";
        end

        % fit
        try
            g = fitglm(T, f, 'Distribution', 'normal');
        catch ME
            fprintf('fitglm failed | model=%s | bin=%d\n', modelNames{m}, k);
            fprintf('%s\n', ME.message);
            continue;
        end

        Fitted_models(k, m).g = g;

        AIC_mat(k, m)  = g.ModelCriterion.AIC;
        BIC_mat(k, m)  = g.ModelCriterion.BIC;
        Nobs_mat(k, m) = sum(mask);

        coefNames = string(g.CoefficientNames);
        coefEst   = g.Coefficients.Estimate;
        coefSE    = g.Coefficients.SE;

        for tt = 1:numel(coefVarNames)
            nm = coefVarNames(tt);
            hit = find(coefNames == nm, 1, 'first');

            if ~isempty(hit)
                betas(k, tt)    = coefEst(hit);
                beta_ses(k, tt) = coefSE(hit);
            end
        end
    end

    Models(m).name         = modelNames{m};
    Models(m).labels       = labels;
    Models(m).coefVarNames = coefVarNames;
    Models(m).betas        = betas;
    Models(m).beta_ses     = beta_ses;
end

end