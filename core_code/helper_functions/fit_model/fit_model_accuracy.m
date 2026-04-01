function [Models, Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = ...
    fit_model_accuracy(cfg)

% make sure all vectors are columns
minN    = cfg.minN;
ConfY   = cfg.ConfY;
Correct = cfg.Correct;
z_coh   = cfg.z_coh;
z_perf  = cfg.z_perf;
rtX     = cfg.rtX;
subjID  = cfg.subjID;
resVol  = cfg.resVol;
z_cond  = cfg.z_cond;
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
    % -------------------------------------------------

    labels = {'b0 (Intercept)', 'b_{rt}', 'b_{coh}', 'b_{cond}'};
    coefVarNames = ["(Intercept)", "R", "coh", "z_cond"];
    baseFormula = "C ~ 1 + R + coh + z_cond";

    % optional one-way terms

    if modelSpec(m).use1(1)
        labels{end+1} = 'b_{vol}';
        coefVarNames(end+1) = "V";
    end

    if modelSpec(m).use2(1)
        labels{end+1} = 'b_{volxcoh}';
        coefVarNames(end+1) = "Vxcoh";
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
        z_cond1 = z_cond(mask);

        % interactions
        Vxcoh = V.*coh;


        % table
        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);

            T = table(C, R, coh, z_cond1, V, Vxcoh, S2, S3, ...
                'VariableNames', {'C','R','coh','z_cond','V', ...
                                  'Vxcoh', 'S2','S3'});
        else
            T = table(C, R, coh, z_cond1, V, Vxcoh, ...
                'VariableNames', {'C','R','coh','z_cond','V', ...
                                  'Vxcoh'});
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

        if useSubjDummies
            f = f + " + S2 + S3";
        end

        % fit
        try
            g = fitglm(T, f, 'Distribution', 'binomial', 'Link', 'logit');
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