function [AIC_mat, BIC_mat, Pool, Fitted_models] = fit_models_pooled(idxSplit, cfg)

% fit_models_pooled - fits family of models to data passed in by cfg file
%
% outputs:
% AIC_mat: a structure containing AIC values for each fitted model at each timebin
% BIC_mat:a structure containing BIC values for each fitted model at each timebin
% Pool: VANESSA HW: FILL THIS IN
% Fitted_models: a structure containing fitted model objects at each timebin
%
% -----
% inputs:
% idxSplit: boolean vector indicating which trials to include
% cfg: configuration file with outcome, predictors, and model spec variables

%%
% unpack config
ConfY = cfg.ConfY;
Correct = cfg.Correct;
subjID = cfg.subjID;
p_perf_online = cfg.p_perf_online;
RTz_all = cfg.RTz_all;
resVol_mat = cfg.resVol_mat;
modelNames = cfg.modelNames;
modelSpec = cfg.modelSpec;
baseLabels = cfg.baseLabels;
twoWayNames = cfg.twoWayNames;
threeWayNames = cfg.threeWayNames;
fourWayNames = cfg.fourWayNames;
useSubjDummies = cfg.useSubjDummies;
minN_pooled = cfg.minN_pooled;

% set up structures for storing data
K = size(resVol_mat,2);
nModels = numel(modelNames);

AIC_mat = nan(K, nModels);
BIC_mat = nan(K, nModels);
Pool    = struct();
Fitted_models = struct();

% labels used for pooled coefficient storage
twoWayLabels   = {'b_{P×C}','b_{P×V}','b_{P×R}','b_{V×C}','b_{C×R}','b_{R×V}'};
threeWayLabels = {'b_{P×V×C}','b_{P×C×R}','b_{P×V×R}','b_{V×C×R}'};
fourWayLabels  = {'b_{P×V×C×R}'};

for m = 1:nModels
    fprintf('\n=== [%s] Pooled fit: %s ===\n', 'SPLIT', modelNames{m});

    % build term list for this model
    termLabels = baseLabels;
    termNames  = ["(Intercept)","perf","corr","vol","rt"];

    for j = 1:6
        if modelSpec(m).use2(j)
            termLabels{end+1} = twoWayLabels{j}; %#ok<AGROW>
            termNames(end+1)  = twoWayNames(j);  %#ok<AGROW>
        end
    end
    for j = 1:4
        if modelSpec(m).use3(j)
            termLabels{end+1} = threeWayLabels{j}; %#ok<AGROW>
            termNames(end+1)  = threeWayNames(j);  %#ok<AGROW>
        end
    end
    if modelSpec(m).use4
        termLabels{end+1} = fourWayLabels{1}; %#ok<AGROW>
        termNames(end+1)  = fourWayNames;     %#ok<AGROW>
    end

    nTerms = numel(termNames);
    beta_pool = nan(K, nTerms);
    se_pool   = nan(K, nTerms);
    p_pool    = nan(K, nTerms);

    for k = 1:K
        Vk = resVol_mat(:,k);

        if sum(idxSplit) < minN_pooled
            continue;
        end

        % split into early / late using idxSplit
        y    = ConfY(idxSplit);
        P    = p_perf_online(idxSplit);
        C    = Correct(idxSplit);
        Vraw = Vk(idxSplit);
        R    = RTz_all(idxSplit);
        sID  = subjID(idxSplit);

        sv = std(Vraw);
        if isnan(sv) || sv < 1e-12
            continue;
        end
        V = (Vraw - mean(Vraw)) ./ sv;

        % interactions
        PxC = P.*C; PxV = P.*V; PxR = P.*R;
        VxC = V.*C; CxR = C.*R; RxV = R.*V;

        PxVxC   = P.*V.*C;
        PxCxR   = P.*C.*R;
        PxVxR   = P.*V.*R;
        VxCxR   = V.*C.*R;
        PxVxCxR = P.*V.*C.*R;

        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);

            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, ...
                PxVxC,PxCxR,PxVxR,VxCxR,PxVxCxR,S2,S3, ...
                'VariableNames', {'confY','perf','corr','vol','rt', ...
                'PxC','PxV','PxR','VxC','CxR','RxV', ...
                'PxVxC','PxCxR','PxVxR','VxCxR','PxVxCxR','S2','S3'});
        else
            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, ...
                PxVxC,PxCxR,PxVxR,VxCxR,PxVxCxR, ...
                'VariableNames', {'confY','perf','corr','vol','rt', ...
                'PxC','PxV','PxR','VxC','CxR','RxV', ...
                'PxVxC','PxCxR','PxVxR','VxCxR','PxVxCxR'});
        end

        f = "confY ~ perf + corr + vol + rt";
        for j = 1:6
            if modelSpec(m).use2(j)
                f = f + " + " + twoWayNames(j);
            end
        end
        for j = 1:4
            if modelSpec(m).use3(j)
                f = f + " + " + threeWayNames(j);
            end
        end
        if modelSpec(m).use4
            f = f + " + " + fourWayNames;
        end
        if useSubjDummies
            f = f + " + S2 + S3";
        end

        try
            g = fitglm(T, f, 'Distribution','normal', 'Link','identity');
        catch
            continue;
        end

        AIC_mat(k,m) = g.ModelCriterion.AIC;
        BIC_mat(k,m) = g.ModelCriterion.BIC;
        Fitted_models(k, m).g = g;

        % store pooled coefficients from THIS fit
        coefNames = string(g.CoefficientNames);
        coefEst   = g.Coefficients.Estimate;
        coefSE    = g.Coefficients.SE;
        coefP     = g.Coefficients.pValue;

        for tt = 1:nTerms
            nm = termNames(tt);
            hit = find(coefNames == nm, 1, 'first');
            if ~isempty(hit)
                beta_pool(k,tt) = coefEst(hit);
                se_pool(k,tt)   = coefSE(hit);
                p_pool(k,tt)    = coefP(hit);
            end
        end
    end

    Pool(m).mName      = modelNames{m};
    Pool(m).termLabels = termLabels;
    Pool(m).termNames  = termNames;
    Pool(m).beta_pool  = beta_pool;
    Pool(m).se_pool    = se_pool;
    Pool(m).p_pool     = p_pool;
end

end