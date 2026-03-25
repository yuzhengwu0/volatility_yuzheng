function Sel = refit_models_perSubjectPerBin( ...
    idxSplit, ConfY, Correct, subjID, split_perf, Cz_all, RTz_all, resVol_time, t_norm, ...
    modelNames, modelSpec, baseLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
    topIdx, minN_sub)

subj_list = unique(subjID(~isnan(subjID)))';
nSubj = numel(subj_list);
K = size(resVol_time, 2);

Sel = struct();

for ii = 1:numel(topIdx)
    mIdx  = topIdx(ii);
    mName = modelNames{mIdx};

    termLabels = baseLabels;
    termNames  = ["(Intercept)","perf","corr","vol","rt"];

    for j = 1:6
        if modelSpec(mIdx).use2(j)
            termLabels{end+1} = twoWayLabels{j}; %#ok<AGROW>
            termNames(end+1)  = twoWayNames(j);  %#ok<AGROW>
        end
    end
    for j = 1:4
        if modelSpec(mIdx).use3(j)
            termLabels{end+1} = threeWayLabels{j}; %#ok<AGROW>
            termNames(end+1)  = threeWayNames(j);  %#ok<AGROW>
        end
    end
    if modelSpec(mIdx).use4
        termLabels{end+1} = fourWayLabels{1}; %#ok<AGROW>
        termNames(end+1)  = fourWayNames;     %#ok<AGROW>
    end

    nTerms   = numel(termNames);
    minN_eff = max(minN_sub, nTerms + 2);   % important

    beta_sub = nan(nSubj, K, nTerms);
    se_sub   = nan(nSubj, K, nTerms);
    p_sub    = nan(nSubj, K, nTerms);

    fprintf('\n--- Refit (per subject/bin): %s ---\n', mName);
    fprintf('nTerms = %d, minN_eff = %d\n', nTerms, minN_eff);

    nFitOK   = 0;
    nTooFew  = 0;
    nZeroStd = 0;
    nFitErr  = 0;
    shownErr = 0;

    for iSub = 1:nSubj
        s = subj_list(iSub);

        for k = 1:K
            Vk = resVol_time(:,k);

            mask = idxSplit ...
                & (subjID == s) ...
                & ~isnan(Vk) ...
                & ~isnan(ConfY) ...
                & ~isnan(Correct) ...
                & ~isnan(split_perf) ...
                & ~isnan(Cz_all) ...
                & ~isnan(RTz_all);

            nMask = sum(mask);

            if nMask < minN_eff
                nTooFew = nTooFew + 1;
                continue;
            end

            y    = ConfY(mask);
            P    = split_perf(mask);
            C    = Cz_all(mask);
            Vraw = Vk(mask);
            R    = RTz_all(mask);

            sv = std(Vraw);
            if isnan(sv) || sv < 1e-12
                nZeroStd = nZeroStd + 1;
                continue;
            end
            V = (Vraw - mean(Vraw)) ./ sv;

            % optional: skip obviously constant predictors
            if std(P) < 1e-12 || std(R) < 1e-12
                nZeroStd = nZeroStd + 1;
                continue;
            end

            % interactions
            PxC = P.*C; PxV = P.*V; PxR = P.*R;
            VxC = V.*C; CxR = C.*R; RxV = R.*V;

            PxVxC   = P.*V.*C;
            PxCxR   = P.*C.*R;
            PxVxR   = P.*V.*R;
            VxCxR   = V.*C.*R;
            PxVxCxR = P.*V.*C.*R;

            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, ...
                PxVxC,PxCxR,PxVxR,VxCxR,PxVxCxR, ...
                'VariableNames', {'confY','perf','corr','vol','rt', ...
                'PxC','PxV','PxR','VxC','CxR','RxV', ...
                'PxVxC','PxCxR','PxVxR','VxCxR','PxVxCxR'});

            f = "confY ~ perf + corr + vol + rt";
            for j = 1:6
                if modelSpec(mIdx).use2(j)
                    f = f + " + " + twoWayNames(j);
                end
            end
            for j = 1:4
                if modelSpec(mIdx).use3(j)
                    f = f + " + " + threeWayNames(j);
                end
            end
            if modelSpec(mIdx).use4
                f = f + " + " + fourWayNames;
            end

            try
                g = fitglm(T, f, 'Distribution','normal', 'Link','identity');
                nFitOK = nFitOK + 1;
            catch ME
                nFitErr = nFitErr + 1;
                if shownErr < 5
                    fprintf('fitglm failed | model=%s | sub=%g | bin=%d | n=%d\n', ...
                        mName, s, k, nMask);
                    fprintf('Reason: %s\n', ME.message);
                    shownErr = shownErr + 1;
                end
                continue;
            end

            coefNames = string(g.CoefficientNames);
            coefEst   = g.Coefficients.Estimate;
            coefSE    = g.Coefficients.SE;
            coefP     = g.Coefficients.pValue;

            for tt = 1:nTerms
                nm = termNames(tt);
                hit = find(coefNames == nm, 1, 'first');
                if ~isempty(hit)
                    beta_sub(iSub,k,tt) = coefEst(hit);
                    se_sub(iSub,k,tt)   = coefSE(hit);
                    p_sub(iSub,k,tt)    = coefP(hit);
                end
            end
        end
    end

    fprintf('Summary for %s: fitOK=%d | tooFew=%d | zeroStd=%d | fitErr=%d\n', ...
        mName, nFitOK, nTooFew, nZeroStd, nFitErr);

    Sel(ii).mName      = mName;
    Sel(ii).termLabels = termLabels;
    Sel(ii).termNames  = termNames;
    Sel(ii).beta_sub   = beta_sub;
    Sel(ii).se_sub     = se_sub;
    Sel(ii).p_sub      = p_sub;
end

end