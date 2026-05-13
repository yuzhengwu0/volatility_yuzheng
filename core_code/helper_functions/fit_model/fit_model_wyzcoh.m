function [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = ...
    fit_model_wyzcoh(cfg)

% make sure all vectors are columns
minN    = cfg.minN;
ConfY   = cfg.ConfY;
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


    labels = {'b0 (Intercept)', 'b_{rt}'};
    coefVarNames = ["(Intercept)", "R"];
    baseFormula = "ConfY ~ 1 + R";

    % optional one-way terms

    if modelSpec(m).use1
        labels{end+1} = 'b_{vol}';
        coefVarNames(end+1) = "V";
    end



    nTerms   = numel(labels);
    betas    = nan(K, nTerms);
    beta_ses = nan(K, nTerms);

    for k = 1:K
        Vk = vol(:, k);

        mask = ~isnan(Vk)      & ...
               ~isnan(ConfY)   & ...
               ~isnan(rtX)     & ...
               ~isnan(subjID);

        if sum(mask) < minN
            continue;
        end

        % current-bin data
        y   = ConfY(mask);
        R   = rtX(mask);
        V   = Vk(mask);
        sID = subjID(mask);


        % table
        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);

            T = table(y, R, V, S2, S3, ...
                'VariableNames', {'ConfY','R','V', 'S2','S3'});
        else
            T = table(y, R, V, ...
                'VariableNames', {'ConfY','R','V'});
        end

        % -------------------------------------------------
        % Build formula
        % -------------------------------------------------
        f = baseFormula;

       if modelSpec(m).use1
            f = f + " + V";
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