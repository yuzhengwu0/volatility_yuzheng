function Sel = refit_top_models_by_subj_coh(cfg, Models)

% ===================== unpack cfg =====================
ConfY      = cfg.ConfY;
Correct    = cfg.Correct;
z_coh      = cfg.z_coh;
z_perf     = cfg.z_perf;
rtX        = cfg.rtX;
subjID     = cfg.subjID;
resVol     = cfg.resVol;
modelNames = cfg.modelNames;
modelSpec  = cfg.modelSpec;
top4Idx    = cfg.top4Idx;

minN_sub = 5;
sv_tol = 1e-12;


% ===================== basic sizes =====================
subj_list = unique(subjID(:))';
nSubj     = numel(subj_list);
[~, K]    = size(resVol);

modelIdxToRefit = top4Idx(:)';

Sel = struct();


% ===================== loop over selected models =====================
for ii = 1:numel(modelIdxToRefit)
    mIdx  = modelIdxToRefit(ii);
    mName = modelNames{mIdx};

    % -------------------------------------------------
    % Use same term definitions as pooled model
    % -------------------------------------------------
    if modelSpec(mIdx).use3
        termLabels = {'b0 (Intercept)', 'b_{corr}', 'b_{rt}', 'b_{perf}', ...
                      'b_{vol}', 'b_{perf\times vol}', 'b_{perf\times vol\times coh}'};
        coefVarNames = ["(Intercept)", "C", "R", "P", "V", "PxV", "PxVxcoh"];
        baseFormula = "ConfY ~ 1 + C + R";
    else
        termLabels = {'b0 (Intercept)', 'b_{corr}', 'b_{rt}', 'b_{coh}', ...
                      'b_{perf}', 'b_{vol}', 'b_{perf\times vol}'};
        coefVarNames = ["(Intercept)", "C", "R", "coh", "P", "V", "PxV"];
        baseFormula = "ConfY ~ 1 + C + R + coh";
    end

    % keep only terms actually used by this model
    keepMask = false(size(coefVarNames));

    keepMask(coefVarNames == "(Intercept)") = true;
    keepMask(coefVarNames == "C") = true;
    keepMask(coefVarNames == "R") = true;

    if ~modelSpec(mIdx).use3
        keepMask(coefVarNames == "coh") = true;
    end

    if modelSpec(mIdx).use1(1)
        keepMask(coefVarNames == "P") = true;
    end

    if modelSpec(mIdx).use1(2)
        keepMask(coefVarNames == "V") = true;
    end

    if modelSpec(mIdx).use2(1)
        keepMask(coefVarNames == "PxV") = true;
    end

    if modelSpec(mIdx).use3
        keepMask(coefVarNames == "PxVxcoh") = true;
    end

    termLabels   = termLabels(keepMask);
    coefVarNames = coefVarNames(keepMask);

    nTerms   = numel(coefVarNames);
    beta_sub = nan(nSubj, K, nTerms);
    se_sub   = nan(nSubj, K, nTerms);
    p_sub    = nan(nSubj, K, nTerms);

    fprintf('\n--- Refit %s per subject/per bin ---\n', mName);

    % -------------------------------------------------
    % Build formula once for this model
    % -------------------------------------------------
    f = baseFormula;

    if modelSpec(mIdx).use1(1)
        f = f + " + P";
    end

    if modelSpec(mIdx).use1(2)
        f = f + " + V";
    end

    if modelSpec(mIdx).use2(1)
        f = f + " + PxV";
    end

    if modelSpec(mIdx).use3
        f = f + " + PxVxcoh";
    end

    % -------------------------------------------------
    % Refit per subject x per bin
    % -------------------------------------------------
    for iSub = 1:nSubj
        thisSub = subj_list(iSub);

        for k = 1:K
            Vk = resVol(:, k);

            mask = ~isnan(Vk)      & ...
                   ~isnan(ConfY)   & ...
                   ~isnan(Correct) & ...
                   ~isnan(z_coh)   & ...
                   ~isnan(z_perf)  & ...
                   ~isnan(rtX)     & ...
                   ~isnan(subjID);

            mask = mask & (subjID == thisSub);

            if sum(mask) < minN_sub
                continue;
            end

            y   = ConfY(mask);
            C   = Correct(mask);
            R   = rtX(mask);
            coh = z_coh(mask);
            P   = z_perf(mask);
            V   = Vk(mask);

            if std(V, 'omitnan') < sv_tol
                continue;
            end

            PxV     = P .* V;
            PxVxcoh = P .* V .* coh;

            T = table(y, C, R, coh, P, V, PxV, PxVxcoh, ...
                'VariableNames', {'ConfY','C','R','coh','P','V','PxV','PxVxcoh'});

            try
                g = fitglm(T, f, 'Distribution', 'normal');
            catch
                continue;
            end

            coefNames = string(g.CoefficientNames);
            coefEst   = g.Coefficients.Estimate;
            coefSE    = g.Coefficients.SE;
            coefP     = g.Coefficients.pValue;

            for tt = 1:nTerms
                nm = coefVarNames(tt);
                hit = find(coefNames == nm, 1, 'first');

                if ~isempty(hit)
                    beta_sub(iSub, k, tt) = coefEst(hit);
                    se_sub(iSub, k, tt)   = coefSE(hit);
                    p_sub(iSub, k, tt)    = coefP(hit);
                end
            end
        end
    end

    % ===================== save into Sel =====================
    Sel(ii).mIdx         = mIdx;
    Sel(ii).mName        = mName;
    Sel(ii).subj_list    = subj_list;
    Sel(ii).termLabels   = termLabels;
    Sel(ii).coefVarNames = coefVarNames;
    Sel(ii).beta_sub     = beta_sub;
    Sel(ii).se_sub       = se_sub;
    Sel(ii).p_sub        = p_sub;

    Sel(ii).beta_pool    = Models(mIdx).betas;
    Sel(ii).se_pool      = Models(mIdx).beta_ses;
end

end