%% regression_with_rt_newFamily.m
% Goals:
% 1) RPF per subject -> predicted performance p_perf(j)
% 2) motion_energy -> time-resolved residual volatility (trial x time)
% 3) Logistic regression at each time bin with NEW model family (including RT):
%       conf ~ perf + corr + vol + rt + interactions (by model)
% 4) Select models by AIC/BIC within THIS family
% 5) Plot BIG FIGURE for selected models (top by BIC by default)

clear; clc;

%% PLOT SWITCHES
DO_PLOT_BIG_FIGURE = true;
DO_PLOT_SLOPE_BIG_FIGURE = true; 
SLOPE_WIN = 5;

%% 0. Add toolboxes
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/Palamedes1'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/boundedline-pkg-master'));
RPF_check_toolboxes;

%% 1. Load data & basic fields
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh_all       = allStruct.rdm1_coh(:);
resp_all      = allStruct.req_resp(:);        % 1/2
correct_all   = allStruct.correct(:);         % 1/0
confCont_all  = allStruct.confidence(:);      % 0-1
vol_all       = allStruct.rdm1_coh_std(:);
subjID_all    = allStruct.group(:);
ME_cell_all   = allStruct.motion_energy;
rt_all        = allStruct.rt(:);              % NEW

valid = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
        ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all);

coh           = coh_all(valid);
resp          = resp_all(valid);
Correct       = correct_all(valid);
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = ME_cell_all(valid);
rt            = rt_all(valid);

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% Binary confidence
th   = 0.5;
Conf = double(confCont >= th);  % high=1 low=0

%% 2. Map volatility to condition index (for RPF)
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;

%% 3. RPF -> predicted performance p_perf_all
subj_list  = unique(subjID);
nSubj      = numel(subj_list);
p_perf_all = nan(nTrials, 1);

for iSub = 1:nSubj
    thisSub = subj_list(iSub);
    fprintf('\n=== Running RPF for subject %d ===\n', thisSub);

    idxS = (subjID == thisSub);

    coh_s     = coh(idxS);
    resp_s    = resp(idxS);
    correct_s = Correct(idxS);
    conf_s    = confCont(idxS);
    cond_s    = cond(idxS);

    if isempty(coh_s), continue; end
    nTr = numel(coh_s);

    resp01 = resp_s - 1; % 1->0, 2->1

    stim01 = resp01;
    wrong_idx = (correct_s == 0);
    stim01(wrong_idx) = 1 - resp01(wrong_idx);

    conf_clip = min(max(conf_s,0),1);
    edges     = [0, 0.25, 0.5, 0.75, 1];
    rating_s  = discretize(conf_clip, edges, 'IncludedEdge', 'right');
    rating_s(isnan(rating_s)) = 4;

    trialData = struct();
    trialData.stimID    = stim01(:)';
    trialData.response  = resp01(:)';
    trialData.rating    = rating_s(:)';
    trialData.correct   = correct_s(:)';
    trialData.x         = coh_s(:)';
    trialData.condition = cond_s(:)';
    trialData.RT        = nan(1,nTr);

    F1 = struct();
    F1.info.DV                     = 'd''';
    F1.info.PF                     = @RPF_scaled_Weibull;
    F1.info.padCells               = 1;
    F1.info.set_P_max_to_d_pad_max = 1;
    F1.info.x_min                  = 0;
    F1.info.x_max                  = 1;
    F1.info.x_label                = 'coherence';
    F1.info.cond_labels            = {'low volatility','high volatility'};
    F1 = RPF_get_F(F1.info, trialData);

    p_perf_trial = nan(nTr,1);
    nCond = numel(F1.data);

    for c = 1:nCond
        mask_c = (cond_s == c);
        if ~any(mask_c), continue; end

        coh_c  = coh_s(mask_c);
        x_grid = F1.data(c).x(:);
        d_grid = F1.data(c).P(:);

        [~, loc] = ismember(coh_c, x_grid);
        d_pred = nan(size(coh_c));
        ok = (loc > 0);
        d_pred(ok) = d_grid(loc(ok));
        if any(~ok)
            d_pred(~ok) = interp1(x_grid, d_grid, coh_c(~ok), 'linear', 'extrap');
        end

        p_corr = normcdf(d_pred ./ sqrt(2));
        p_perf_trial(mask_c) = p_corr;
    end

    p_perf_all(idxS) = p_perf_trial;
end

fprintf('Finished RPF. Valid p_perf proportion: %.3f\n', mean(~isnan(p_perf_all)));

%% 4. Compute residual volatility from motion_energy
winLen = 10;
tol    = 1e-12;

evidence_strength   = cell(nTrials, 1);
volatility_strength = cell(nTrials, 1);

for tr = 1:nTrials
    frames = motion_energy{tr};
    trace  = frames(:)';

    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);

    if nFrames < winLen
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    nWin  = nFrames - winLen + 1;
    m_win = nan(1, nWin);
    s_win = nan(1, nWin);

    for w = 1:nWin
        seg      = trace_eff(w : w + winLen - 1);
        m_win(w) = mean(seg);
        s_win(w) = std(seg);
    end

    evidence_strength{tr}   = m_win;
    volatility_strength{tr} = s_win;
end

nBins  = 40;
t_norm = linspace(0, 1, nBins);

MEAN_norm = nan(nTrials, nBins);
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};
    if isempty(mu_tr) || isempty(sd_tr), continue; end

    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    t_orig = linspace(0, 1, nWin_tr);
    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 3, continue; end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    resid = y_use - Xb*beta;

    tmp = nan(size(y));
    tmp(mask_b) = resid;
    resVol_mat(:, b) = tmp;
end

mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_time = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d bins\n', size(resVol_time,1), size(resVol_time,2));

%% 5. Build predictors: perf/corr/rt
eps0   = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);
Fp_all = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

Cz_all = Correct - mean(Correct,'omitnan');

% RT refine (log + global z)
rt_eps  = 1e-6;
rt_ref  = log(rt + rt_eps);   % log-RT
RTz_all = (rt_ref - mean(rt_ref,'omitnan')) ./ std(rt_ref,'omitnan');

%% 6. NEW MODEL FAMILY + fit per bin (AIC/BIC + betas)
[N, K] = size(resVol_time);
minN = 50;
useSubjDummies = true;

% --- Define term labels ---
baseLabels = {'b0 (Intercept)','b_{perf}','b_{corr}','b_{vol}','b_{rt}'};

twoWayNames  = ["PxC","PxV","PxR","VxC","CxR","RxV"];
twoWayLabels = {'b_{perf×corr}','b_{perf×vol}','b_{perf×rt}', ...
                'b_{vol×corr}','b_{corr×rt}','b_{rt×vol}'};

threeWayNames  = ["PxVxC","PxCxR","PxVxR","VxCxR"];
threeWayLabels = {'b_{perf×vol×corr}','b_{perf×corr×rt}', ...
                  'b_{perf×vol×rt}','b_{vol×corr×rt}'};

fourWayNames  = "PxVxCxR";
fourWayLabels = {'b_{perf×vol×corr×rt}'};

% --- Build model list (NEW family) ---
modelNames = {};
modelSpec  = struct('use2',{},'use3',{},'use4',{});

% M0: baseline only
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M0_base';
modelSpec(idx).use2  = false(1,6);
modelSpec(idx).use3  = false(1,4);
modelSpec(idx).use4  = false;

% M1: baseline + perf*corr (PxC)
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M1_PC';
modelSpec(idx).use2  = false(1,6); 
modelSpec(idx).use2(1) = true;     % PxC
modelSpec(idx).use3  = false(1,4);
modelSpec(idx).use4  = false;

% M2–M6: add one 2-way at a time (excluding PxC)
oneAtATime = [5 6 2 3 4];  % CxR, RxV, PxV, PxR, VxC

for ii = 1:numel(oneAtATime)
    j = oneAtATime(ii);
    idx = numel(modelNames) + 1;
    modelNames{idx}     = sprintf('M%d_2way_%s', 1+ii, twoWayNames(j));
    modelSpec(idx).use2 = false(1,6);
    modelSpec(idx).use2(j) = true;
    modelSpec(idx).use3 = false(1,4);
    modelSpec(idx).use4 = false;
end

% M7: all 2-way
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M7_all2';
modelSpec(idx).use2  = true(1,6);
modelSpec(idx).use3  = false(1,4);
modelSpec(idx).use4  = false;

% M8: all 2-way + all 3-way
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M8_all2_all3';
modelSpec(idx).use2  = true(1,6);
modelSpec(idx).use3  = true(1,4);
modelSpec(idx).use4  = false;

% M9: full (2-way + 3-way + 4-way)
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M9_full';
modelSpec(idx).use2  = true(1,6);
modelSpec(idx).use3  = true(1,4);
modelSpec(idx).use4  = true;

nModels = numel(modelNames);

AIC_mat = nan(K, nModels);
BIC_mat = nan(K, nModels);
Nobs_mat= nan(K, nModels);

Models = struct();

for m = 1:nModels
    fprintf('\n=== Fitting %s ===\n', modelNames{m});

    % Build labels + coefVarNames for THIS model
    labels = baseLabels;
    coefVarNames = ["(Intercept)","perf","corr","vol","rt"];

    % 2-way
    for j = 1:6
        if modelSpec(m).use2(j)
            labels{end+1} = twoWayLabels{j};
            coefVarNames(end+1) = twoWayNames(j);
        end
    end
    % 3-way
    for j = 1:4
        if modelSpec(m).use3(j)
            labels{end+1} = threeWayLabels{j};
            coefVarNames(end+1) = threeWayNames(j);
        end
    end
    % 4-way
    if modelSpec(m).use4
        labels{end+1} = fourWayLabels{1};
        coefVarNames(end+1) = fourWayNames;
    end

    nTerms = numel(labels);
    betas  = nan(K, nTerms);

    for k = 1:K
        Vk = resVol_time(:,k);

        mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(subjID) & ~isnan(RTz_all);
        if sum(mask) < minN, continue; end

        y    = Conf(mask);
        P    = Fp_all(mask);
        C    = Cz_all(mask);
        Vraw = Vk(mask);
        R    = RTz_all(mask);
        sID  = subjID(mask);

        sv = std(Vraw);
        if sv < 1e-12, continue; end
        V = (Vraw - mean(Vraw)) ./ sv;

        % all interactions
        PxC = P.*C; PxV = P.*V; PxR = P.*R;
        VxC = V.*C; CxR = C.*R; RxV = R.*V;

        PxVxC = P.*V.*C;
        PxCxR = P.*C.*R;
        PxVxR = P.*V.*R;
        VxCxR = V.*C.*R;

        PxVxCxR = P.*V.*C.*R;

        % build table
        if useSubjDummies
            S2 = double(sID==2);
            S3 = double(sID==3);
            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, S2,S3, ...
                'VariableNames', {'conf','perf','corr','vol','rt', ...
                                  'PxC','PxV','PxR','VxC','CxR','RxV', ...
                                  'PxVxC','PxCxR','PxVxR','VxCxR', ...
                                  'PxVxCxR','S2','S3'});
        else
            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, ...
                'VariableNames', {'conf','perf','corr','vol','rt', ...
                                  'PxC','PxV','PxR','VxC','CxR','RxV', ...
                                  'PxVxC','PxCxR','PxVxR','VxCxR', ...
                                  'PxVxCxR'});
        end

        % formula
        f = "conf ~ perf + corr + vol + rt";
        for j = 1:6
            if modelSpec(m).use2(j), f = f + " + " + twoWayNames(j); end
        end
        for j = 1:4
            if modelSpec(m).use3(j), f = f + " + " + threeWayNames(j); end
        end
        if modelSpec(m).use4, f = f + " + " + fourWayNames; end
        if useSubjDummies, f = f + " + S2 + S3"; end

        try
            g = fitglm(T, f, 'Distribution','binomial', 'Link','logit');
        catch
            continue;
        end

        AIC_mat(k,m)  = g.ModelCriterion.AIC;
        BIC_mat(k,m)  = g.ModelCriterion.BIC;
        Nobs_mat(k,m) = sum(mask);

        coefNames = string(g.CoefficientNames);
        coefEst   = g.Coefficients.Estimate;

        for tt = 1:numel(coefVarNames)
            nm = coefVarNames(tt);
            idx = find(coefNames == nm, 1, 'first');
            if ~isempty(idx)
                betas(k,tt) = coefEst(idx);
            end
        end
    end

    Models(m).name        = modelNames{m};
    Models(m).labels      = labels;
    Models(m).coefVarNames= coefVarNames;
    Models(m).betas       = betas;
end

%% Delta AIC/BIC summary
minAIC_perBin = min(AIC_mat, [], 2, 'omitnan');
minBIC_perBin = min(BIC_mat, [], 2, 'omitnan');

deltaAIC_mat = AIC_mat - minAIC_perBin;
deltaBIC_mat = BIC_mat - minBIC_perBin;

meanDeltaAIC = mean(deltaAIC_mat, 1, 'omitnan');
medDeltaAIC  = median(deltaAIC_mat, 1, 'omitnan');
meanDeltaBIC = mean(deltaBIC_mat, 1, 'omitnan');
medDeltaBIC  = median(deltaBIC_mat, 1, 'omitnan');

deltaTbl = table(modelNames(:), meanDeltaAIC(:), medDeltaAIC(:), ...
                             meanDeltaBIC(:), medDeltaBIC(:), ...
    'VariableNames', {'Model','Mean_dAIC','Median_dAIC','Mean_dBIC','Median_dBIC'});

disp('=== Delta AIC/BIC summary (NEW family) ===');
disp(deltaTbl);

%% 8. Choose models for Fig1 & Fig2
score = mean([ ...
    meanDeltaAIC(:), ...
    medDeltaAIC(:), ...
    meanDeltaBIC(:), ...
    medDeltaBIC(:) ...
], 2, 'omitnan');

[~, rankIdx] = sort(score, 'ascend', 'MissingPlacement','last');
rankIdx = rankIdx(~isnan(score(rankIdx)));

bestModelIdx = rankIdx(1);

N_TOP = 4;
top4Idx = rankIdx(1:min(N_TOP, numel(rankIdx)));

modelIdxToRefit = unique([bestModelIdx; top4Idx(:)]);

fprintf('\n=== Best model (Fig1) ===\n');
disp(table(modelNames(bestModelIdx)', score(bestModelIdx), ...
    'VariableNames', {'Model','CompositeScore'}));

fprintf('\n=== Top4 models (Fig2) ===\n');
disp(table(modelNames(top4Idx)', score(top4Idx), ...
    'VariableNames', {'Model','CompositeScore'}));

%% 9) Refit per subject per bin for selected models
subj_list = unique(subjID(:))';
nSubj = numel(subj_list);
K = numel(t_norm);

colSub = [0 0.4470 0.7410; 1 0 0; 0.9290 0.6940 0.1250];
minN_sub = 5;
sv_tol   = 1e-12;

Sel = struct();

for ii = 1:numel(modelIdxToRefit)
    mIdx  = modelIdxToRefit(ii);
    mName = modelNames{mIdx};

    termLabels = baseLabels;
    termNames  = ["(Intercept)","perf","corr","vol","rt"];

    for j = 1:6
        if modelSpec(mIdx).use2(j)
            termLabels{end+1} = twoWayLabels{j};
            termNames(end+1)  = twoWayNames(j);
        end
    end
    for j = 1:4
        if modelSpec(mIdx).use3(j)
            termLabels{end+1} = threeWayLabels{j};
            termNames(end+1)  = threeWayNames(j);
        end
    end
    if modelSpec(mIdx).use4
        termLabels{end+1} = fourWayLabels{1};
        termNames(end+1)  = fourWayNames;
    end

    nTerms = numel(termNames);

    beta_sub = nan(nSubj, K, nTerms);
    se_sub   = nan(nSubj, K, nTerms);
    p_sub    = nan(nSubj, K, nTerms);

    fprintf('\n--- Refit %s per subject/per bin ---\n', mName);

    for iSub = 1:nSubj
        thisSub = subj_list(iSub);

        for k = 1:K
            Vk = resVol_time(:,k);

            mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(subjID) & ~isnan(rt);
            mask = mask & (subjID == thisSub);

            if sum(mask) < minN_sub, continue; end

            y    = Conf(mask);
            P    = Fp_all(mask);
            C    = Cz_all(mask);
            Vraw = Vk(mask);

            sv = std(Vraw);
            if sv < sv_tol, continue; end
            V = (Vraw - mean(Vraw)) ./ sv;

            RTraw = rt(mask);
            RTref = log(RTraw + 1e-6);
            R = (RTref - mean(RTref,'omitnan')) ./ std(RTref,'omitnan');

            PxC = P.*C; PxV = P.*V; PxR = P.*R;
            VxC = V.*C; CxR = C.*R; RxV = R.*V;

            PxVxC = P.*V.*C;
            PxCxR = P.*C.*R;
            PxVxR = P.*V.*R;
            VxCxR = V.*C.*R;

            PxVxCxR = P.*V.*C.*R;

            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, ...
                'VariableNames', {'conf','perf','corr','vol','rt', ...
                                  'PxC','PxV','PxR','VxC','CxR','RxV', ...
                                  'PxVxC','PxCxR','PxVxR','VxCxR', ...
                                  'PxVxCxR'});

            f = "conf ~ perf + corr + vol + rt";
            for j = 1:6
                if modelSpec(mIdx).use2(j), f = f + " + " + twoWayNames(j); end
            end
            for j = 1:4
                if modelSpec(mIdx).use3(j), f = f + " + " + threeWayNames(j); end
            end
            if modelSpec(mIdx).use4, f = f + " + " + fourWayNames; end

            try
                g = fitglm(T, f, 'Distribution','binomial', 'Link','logit');
            catch
                continue;
            end

            coefNames = string(g.CoefficientNames);
            coefEst   = g.Coefficients.Estimate;
            coefSE    = g.Coefficients.SE;
            coefP     = g.Coefficients.pValue;

            for tt = 1:nTerms
                nm = termNames(tt);
                idx = find(coefNames == nm, 1, 'first');
                if ~isempty(idx)
                    beta_sub(iSub,k,tt) = coefEst(idx);
                    se_sub(iSub,k,tt)   = coefSE(idx);
                    p_sub(iSub,k,tt)    = coefP(idx);
                end
            end
        end
    end

    Sel(ii).mName      = mName;
    Sel(ii).termLabels = termLabels;
    Sel(ii).beta_sub   = beta_sub;
    Sel(ii).se_sub     = se_sub;
    Sel(ii).p_sub      = p_sub;
end

%% 10) EXPORT TWO FIGURES (Fig1 & Fig2)

% ========== FIGURE 1: BEST model -> ALL betas ==========
bestName = modelNames{bestModelIdx};
hitBest  = find(strcmp({Sel.mName}, bestName), 1, 'first');
if isempty(hitBest)
    error('Best model %s not found in Sel. Did Section 9 run?', bestName);
end
SelBest = Sel(hitBest);

outName1 = 'Fig1_BestModel_AllBetas.pdf';
plot_singleModel_allTerms(SelBest, t_norm, colSub, outName1);

% ========== FIGURE 2: TOP4 models -> ONLY volatility-related betas ==========
% top4Names = modelNames(top4Idx);
% outName2  = 'Fig2_Top4_OnlyVolRelatedBetas.pdf';
% plot_multiModel_volTerms_only(Sel, top4Names, t_norm, colSub, outName2);
% 
% fprintf('\n✓ Exported:\n  %s\n  %s\n', outName1, outName2);

%% 11) BIG FIGURE (VOL-ONLY): square tiles + PDF export
if DO_PLOT_BIG_FIGURE
    
    % Use the top4 models for plotting
    modelNamesToPlot = modelNames(top4Idx);
    
    desiredNames = modelNamesToPlot;

    SelOrdered = Sel([]);
    for i = 1:numel(desiredNames)
        hit = find(strcmp({Sel.mName}, desiredNames{i}), 1, 'first');
        if isempty(hit)
            error('Sel missing model %s. Did Section 9 refit run?', desiredNames{i});
        end
        SelOrdered(i) = Sel(hit);
    end
    Sel = SelOrdered;
    clear SelOrdered hit i desiredNames

    masterTermLabels = { ...
        '$b_{\mathrm{vol}}$', ...
        '$b_{\mathrm{perf}\times\mathrm{vol}}$', ...
        '$b_{\mathrm{corr}\times\mathrm{vol}}$', ...
        '$b_{\mathrm{vol}\times\mathrm{rt}}$', ...
        '$b_{\mathrm{perf}\times\mathrm{corr}\times\mathrm{vol}}$', ...
        '$b_{\mathrm{perf}\times\mathrm{vol}\times\mathrm{rt}}$', ...
        '$b_{\mathrm{corr}\times\mathrm{vol}\times\mathrm{rt}}$', ...
        '$b_{\mathrm{perf}\times\mathrm{corr}\times\mathrm{vol}\times\mathrm{rt}}$' ...
    };
    nRows = numel(masterTermLabels);

    modelNamesLatex = cell(1, numel(Sel));
    for m = 1:numel(Sel)
        modelName = Sel(m).mName;
        modelName = strrep(modelName, '_', '\_');
        modelNamesLatex{m} = modelName;
    end

    internalMaster = { ...
        {'b_{vol}'}, ...
        {'b_{perf×vol}','b_{vol×perf}'}, ...
        {'b_{corr×vol}','b_{vol×corr}'}, ...
        {'b_{vol×rt}','b_{rt×vol}'}, ...
        {'b_{perf×corr×vol}','b_{perf×vol×corr}','b_{corr×perf×vol}','b_{corr×vol×perf}','b_{vol×perf×corr}','b_{vol×corr×perf}'}, ...
        {'b_{perf×vol×rt}','b_{perf×rt×vol}','b_{vol×perf×rt}','b_{vol×rt×perf}','b_{rt×perf×vol}','b_{rt×vol×perf}'}, ...
        {'b_{corr×vol×rt}','b_{corr×rt×vol}','b_{vol×corr×rt}','b_{vol×rt×corr}','b_{rt×corr×vol}','b_{rt×vol×corr}'}, ...
        {'b_{perf×corr×vol×rt}','b_{perf×corr×rt×vol}','b_{perf×vol×corr×rt}','b_{perf×vol×rt×corr}', ...
         'b_{perf×rt×corr×vol}','b_{perf×rt×vol×corr}', ...
         'b_{corr×perf×vol×rt}','b_{corr×perf×rt×vol}','b_{corr×vol×perf×rt}','b_{corr×vol×rt×perf}', ...
         'b_{corr×rt×perf×vol}','b_{corr×rt×vol×perf}', ...
         'b_{vol×perf×corr×rt}','b_{vol×perf×rt×corr}','b_{vol×corr×perf×rt}','b_{vol×corr×rt×perf}', ...
         'b_{vol×rt×perf×corr}','b_{vol×rt×corr×perf}', ...
         'b_{rt×perf×corr×vol}','b_{rt×perf×vol×corr}','b_{rt×corr×perf×vol}','b_{rt×corr×vol×perf}', ...
         'b_{rt×vol×perf×corr}','b_{rt×vol×corr×perf}'} ...
    };

    allData = [];
    for r = 1:nRows
        for mIdx = 1:numel(Sel)
            possibleNames = internalMaster{r};
            termList = Sel(mIdx).termLabels;

            tt = [];
            for ii = 1:numel(possibleNames)
                tt = find(strcmp(termList, possibleNames{ii}), 1, 'first');
                if ~isempty(tt), break; end
            end

            if ~isempty(tt)
                beta_sub = Sel(mIdx).beta_sub(:,:,tt);
                se_sub   = Sel(mIdx).se_sub(:,:,tt);
                allData = [allData; beta_sub(:); beta_sub(:)+se_sub(:); beta_sub(:)-se_sub(:)]; %#ok<AGROW>
            end
        end
    end

    allData = allData(~isnan(allData));
    dataMin = min(allData); dataMax = max(allData);

    tickOptions = -3:0.5:3;
    yMin = max(tickOptions(tickOptions <= dataMin));
    yMax = min(tickOptions(tickOptions >= dataMax));

    if isempty(yMin) || isempty(yMax) || (yMax - yMin) < 1.5
        yCenter = (dataMin + dataMax) / 2;
        yMin = floor(yCenter*2)/2 - 0.75;
        yMax = ceil(yCenter*2)/2 + 0.75;
    end

    yLimGlobal   = [yMin, yMax];
    yTicksGlobal = yMin:0.5:yMax;

    fprintf('VOL-only Global Y-axis range: [%.1f, %.1f]\n', yMin, yMax);

    nModelCols = numel(Sel);
    fontPanel = 10;
    lw_sub    = 0.55;
    lw_mean   = 0.90;
    alphaSub  = 0.14;
    alphaMean = 0.08;

    x = t_norm(:)';

    tileSize = 130;
    gapX     = 20;
    gapY     = 20;
    labelW   = 80;

    outerL = 40;
    outerR = 30;
    outerT = 30;
    outerB = 80;

    figW = outerL + labelW + nModelCols*tileSize + (nModelCols-1)*gapX + outerR;
    figH = outerT + nRows*tileSize + (nRows-1)*gapY + outerB;

    fig = figure('Color','w','Units','points','Position',[50 50 figW figH]);

    pt2nx = @(pt) pt / figW;
    pt2ny = @(pt) pt / figH;

    L = pt2nx(outerL);
    R = pt2nx(outerR);
    T = pt2ny(outerT);
    B = pt2ny(outerB);

    gapXNorm   = pt2nx(gapX);
    gapYNorm   = pt2ny(gapY);
    labelWNorm = pt2nx(labelW);

    tileWNorm = pt2nx(tileSize);
    tileHNorm = pt2ny(tileSize);

    usedW = labelWNorm + nModelCols*tileWNorm + (nModelCols-1)*gapXNorm;
    usedH = nRows*tileHNorm + (nRows-1)*gapYNorm;

    x0   = L + ((1 - L - R) - usedW)/2;
    yTop = 1 - T - ((1 - T - B) - usedH)/2;

    lastAxPerModel = gobjects(1, nModelCols);

    for r = 1:nRows
        yPos = yTop - r*tileHNorm - (r-1)*gapYNorm;

        axLab = axes('Parent',fig,'Units','normalized','Position',[x0, yPos, labelWNorm, tileHNorm]);
        axis(axLab,'off');
        text(axLab, 0.50, 0.50, masterTermLabels{r}, ...
            'FontSize', 16, 'FontWeight','bold', ...
            'Rotation', 90, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'Interpreter','latex');

        for mIdx = 1:nModelCols
            xPos = x0 + labelWNorm + (mIdx-1)*(tileWNorm + gapXNorm);

            ax = axes('Parent',fig,'Units','normalized','Position',[xPos, yPos, tileWNorm, tileHNorm]);
            hold(ax,'on'); grid(ax,'on'); box(ax,'off');

            if r == 1
                title(ax, ['$\mathrm{' modelNamesLatex{mIdx} '}$'], ...
                    'Interpreter','latex', 'FontSize', 12, 'FontWeight','bold');
            end

            possibleNames = internalMaster{r};
            termList = Sel(mIdx).termLabels;

            tt = [];
            for ii = 1:numel(possibleNames)
                tt = find(strcmp(termList, possibleNames{ii}), 1, 'first');
                if ~isempty(tt), break; end
            end

            if isempty(tt)
                axis(ax,'off'); set(ax,'Visible','off');
                continue;
            end

            lastAxPerModel(mIdx) = ax;

            beta_sub = Sel(mIdx).beta_sub(:,:,tt);
            se_sub   = Sel(mIdx).se_sub(:,:,tt);

            for iSub = 1:nSubj
                yv = squeeze(beta_sub(iSub,:));
                ev = squeeze(se_sub(iSub,:));

                ok = ~isnan(yv) & ~isnan(ev);
                if sum(ok) >= 2
                    xx = x(ok); yy = yv(ok); ee = ev(ok);
                    fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(iSub,:), ...
                        'EdgeColor','none', 'FaceAlpha', alphaSub, 'HandleVisibility','off');
                end

                plot(ax, x, yv, '-', 'Color', colSub(iSub,:), ...
                    'LineWidth', lw_sub, 'HandleVisibility','off');
             
            end

            yMean = mean(beta_sub, 1, 'omitnan');
            ySEM  = std(beta_sub, 0, 1, 'omitnan') ./ sqrt(nSubj);

            okm = ~isnan(yMean) & ~isnan(ySEM);
            if sum(okm) >= 2
                xx = x(okm); ym = yMean(okm); es = ySEM(okm);
                fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                    'EdgeColor','none', 'FaceAlpha', alphaMean, 'HandleVisibility','off');
            end
            plot(ax, x, yMean, 'k-', 'LineWidth', lw_mean, 'HandleVisibility','off');

            yline(ax, 0, 'k--', 'LineWidth', 0.5, 'HandleVisibility','off');
            xlim(ax, [0 1]);
            ylim(ax, yLimGlobal);
            yticks(ax, yTicksGlobal);
            xticks(ax, 0:0.2:1);
            set(ax, 'FontSize', fontPanel);
            set(ax, 'LineWidth', 0.8);
        end
    end

    for mIdx = 1:nModelCols
        ax = lastAxPerModel(mIdx);
        if ~isempty(ax) && isgraphics(ax)
            xlabel(ax, 'Normalized time (0--1)', 'Interpreter','latex', 'FontSize', 11);
        end
    end

    axLeg = axes('Parent',fig,'Units','normalized','Position',[0.14 0.01 0.60 0.06]);
    axis(axLeg,'off'); hold(axLeg,'on');

    hLeg = gobjects(nSubj+1, 1);
    legText = cell(nSubj+1, 1);

    for s = 1:nSubj
        hLeg(s) = plot(axLeg, nan, nan, '-', 'LineWidth', 2.5, 'Color', colSub(s,:));
        legText{s} = sprintf('$\\mathrm{Subject\\ %d}$', s);
    end
    hLeg(nSubj+1) = plot(axLeg, nan, nan, 'k-', 'LineWidth', 3.0);
    legText{nSubj+1} = '$\mathrm{Mean}$';

    legend(axLeg, hLeg, legText, ...
        'Orientation', 'vertical', ...
        'Location', 'northeast', ...
        'Box', 'off', ...
        'Interpreter', 'latex', ...
        'FontSize', 13);

    outName = 'BigFigure_VOLonly_TermByModel.pdf';

    set(fig,'Renderer','painters');
    set(fig,'PaperUnits','points');
    set(fig,'PaperSize',[figW figH]);
    set(fig,'PaperPosition',[0 0 figW figH]);
    set(fig,'PaperPositionMode','manual');
    set(fig,'PaperOrientation','portrait');

    print(fig, outName, '-dpdf', '-painters');
    fprintf('✓ Saved: %s\n', outName);

end

fprintf('\nDone.\n');

%% 12) BIG FIGURE (SLOPE): same layout as VOL-only Big Figure, but plot d(beta)/dt per bin
if exist('DO_PLOT_SLOPE_BIG_FIGURE','var') && DO_PLOT_SLOPE_BIG_FIGURE

    fprintf('\n=== BIG FIGURE (SLOPE): VOL-only (centered win, keep 3–38 stretched to 0–1) ===\n');

    % ------------------------------------------------------------
    % Reuse:
    %   Sel (top4 models already ordered in your Big Figure block)
    %   masterTermLabels / internalMaster (the 8 volatility-related terms)
    %   colSub, t_norm, nSubj
    % ------------------------------------------------------------

    % ---- slope window ----
    if ~exist('SLOPE_WIN','var') || isempty(SLOPE_WIN)
        SLOPE_WIN = 5;
    end
    if mod(SLOPE_WIN,2)==0
        error('SLOPE_WIN must be odd (e.g., 5,7,9)');
    end
    h = floor(SLOPE_WIN/2);

    % ---- time axis ----
    K = numel(t_norm);          % 40
    idxKeep = (1+h):(K-h);      % 3:38 for win=5
    xFull = t_norm(:)';         % 1x40
    xKeep = linspace(0, 1, numel(idxKeep));  % 36 pts, stretched to 0–1

    slopeFull = @(y) localSlopeSlidingWin(y, xFull, SLOPE_WIN); % returns 1x40

    % ---- dimensions ----
    nModelCols = numel(Sel);
    nRows      = numel(masterTermLabels);

    % ---- gather for global y-range across all panels (using ONLY kept bins) ----
    allData = [];

    for r = 1:nRows
        for mIdx = 1:nModelCols

            possibleNames = internalMaster{r};
            termList = Sel(mIdx).termLabels;

            tt = [];
            for ii = 1:numel(possibleNames)
                tt = find(strcmp(termList, possibleNames{ii}), 1, 'first');
                if ~isempty(tt), break; end
            end
            if isempty(tt), continue; end

            beta_sub = Sel(mIdx).beta_sub(:,:,tt);  % nSubj x K
            se_sub   = Sel(mIdx).se_sub(:,:,tt);    % nSubj x K

            betaSlopeK = nan(nSubj, numel(idxKeep));
            seSlopeK   = nan(nSubj, numel(idxKeep));

            for s = 1:nSubj
                tmpB = slopeFull(beta_sub(s,:));  % 1x40
                tmpE = slopeFull(se_sub(s,:));    % 1x40

                betaSlopeK(s,:) = tmpB(idxKeep);  % 1x36
                seSlopeK(s,:)   = tmpE(idxKeep);  % 1x36
            end

            allData = [allData; betaSlopeK(:); betaSlopeK(:)+seSlopeK(:); betaSlopeK(:)-seSlopeK(:)]; %#ok<AGROW>
        end
    end

    allData = allData(~isnan(allData));
    if isempty(allData)
        warning('Slope figure: allData empty (maybe all NaNs). Skipping slope plot.');
        return;
    end

    dataMin = min(allData);
    dataMax = max(allData);

    % ---- choose nice global ticks (optional) ----
    tickOptions = -8:0.5:8;
    yMin = max(tickOptions(tickOptions <= dataMin));
    yMax = min(tickOptions(tickOptions >= dataMax));

    if isempty(yMin) || isempty(yMax) || (yMax - yMin) < 1.5
        yCenter = (dataMin + dataMax) / 2;
        yMin = floor(yCenter*2)/2 - 0.75;
        yMax = ceil(yCenter*2)/2 + 0.75;
    end

    yLimGlobal   = [yMin, yMax];
    yTicksGlobal = yMin:0.5:yMax;

    fprintf('SLOPE (kept bins) Global Y-axis range: [%.1f, %.1f]\n', yMin, yMax);

    % ---- Plot style ----
    fontPanel = 10;
    lw_sub    = 0.55;
    lw_mean   = 0.90;
    alphaSub  = 0.14;
    alphaMean = 0.08;

    % ---- Layout (same as your Big Figure) ----
    tileSize = 130;
    gapX     = 20;
    gapY     = 20;
    labelW   = 80;

    outerL = 40;
    outerR = 30;
    outerT = 30;
    outerB = 80;

    figW = outerL + labelW + nModelCols*tileSize + (nModelCols-1)*gapX + outerR;
    figH = outerT + nRows*tileSize + (nRows-1)*gapY + outerB;

    fig = figure('Color','w','Units','points','Position',[50 50 figW figH]);

    pt2nx = @(pt) pt / figW;
    pt2ny = @(pt) pt / figH;

    L = pt2nx(outerL);
    R = pt2nx(outerR);
    T = pt2ny(outerT);
    B = pt2ny(outerB);

    gapXNorm   = pt2nx(gapX);
    gapYNorm   = pt2ny(gapY);
    labelWNorm = pt2nx(labelW);

    tileWNorm = pt2nx(tileSize);
    tileHNorm = pt2ny(tileSize);

    usedW = labelWNorm + nModelCols*tileWNorm + (nModelCols-1)*gapXNorm;
    usedH = nRows*tileHNorm + (nRows-1)*gapYNorm;

    x0   = L + ((1 - L - R) - usedW)/2;
    yTop = 1 - T - ((1 - T - B) - usedH)/2;

    % model titles latex-safe
    modelNamesLatex = cell(1, nModelCols);
    for m = 1:nModelCols
        nm = Sel(m).mName;
        nm = strrep(nm, '_', '\_');
        modelNamesLatex{m} = nm;
    end

    lastAxPerModel = gobjects(1, nModelCols);

    % ---- draw panels ----
    for r = 1:nRows
        yPos = yTop - r*tileHNorm - (r-1)*gapYNorm;

        % left label
        axLab = axes('Parent',fig,'Units','normalized','Position',[x0, yPos, labelWNorm, tileHNorm]);
        axis(axLab,'off');
        text(axLab, 0.50, 0.50, masterTermLabels{r}, ...
            'FontSize', 16, 'FontWeight','bold', ...
            'Rotation', 90, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'Interpreter','latex');

        for mIdx = 1:nModelCols
            xPos = x0 + labelWNorm + (mIdx-1)*(tileWNorm + gapXNorm);

            ax = axes('Parent',fig,'Units','normalized','Position',[xPos, yPos, tileWNorm, tileHNorm]);
            hold(ax,'on'); grid(ax,'on'); box(ax,'off');

            if r == 1
                title(ax, ['$\mathrm{' modelNamesLatex{mIdx} '}$'], ...
                    'Interpreter','latex', 'FontSize', 12, 'FontWeight','bold');
            end

            possibleNames = internalMaster{r};
            termList = Sel(mIdx).termLabels;

            tt = [];
            for ii = 1:numel(possibleNames)
                tt = find(strcmp(termList, possibleNames{ii}), 1, 'first');
                if ~isempty(tt), break; end
            end

            if isempty(tt)
                axis(ax,'off'); set(ax,'Visible','off');
                continue;
            end

            lastAxPerModel(mIdx) = ax;

            beta_sub = Sel(mIdx).beta_sub(:,:,tt); % nSubj x 40
            se_sub   = Sel(mIdx).se_sub(:,:,tt);   % nSubj x 40

            % ---- slope (full 40) then keep 36 ----
            betaSlope = nan(nSubj, numel(idxKeep));
            seSlope   = nan(nSubj, numel(idxKeep));
            for s = 1:nSubj
                tmpB = slopeFull(beta_sub(s,:));
                tmpE = slopeFull(se_sub(s,:));
                betaSlope(s,:) = tmpB(idxKeep);
                seSlope(s,:)   = tmpE(idxKeep);
            end

            % ---- plot subjects ----
            for iSub = 1:nSubj
                yv = betaSlope(iSub,:);
                ev = seSlope(iSub,:);

                ok = ~isnan(yv) & ~isnan(ev);
                if sum(ok) >= 2
                    xx = xKeep(ok); yy = yv(ok); ee = ev(ok);
                    fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(iSub,:), ...
                        'EdgeColor','none', 'FaceAlpha', alphaSub, 'HandleVisibility','off');
                end
                plot(ax, xKeep, yv, '-', 'Color', colSub(iSub,:), ...
                    'LineWidth', lw_sub, 'HandleVisibility','off');
            end

            % ---- mean ± SEM ----
            yMean = mean(betaSlope, 1, 'omitnan');
            ySEM  = std(betaSlope, 0, 1, 'omitnan') ./ sqrt(nSubj);

            okm = ~isnan(yMean) & ~isnan(ySEM);
            if sum(okm) >= 2
                xx = xKeep(okm); ym = yMean(okm); es = ySEM(okm);
                fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                    'EdgeColor','none', 'FaceAlpha', alphaMean, 'HandleVisibility','off');
            end
            plot(ax, xKeep, yMean, 'k-', 'LineWidth', lw_mean, 'HandleVisibility','off');

            % ---- y-limits (panel-specific to avoid clipping) ----
            panelData = betaSlope(:);
            panelData = panelData(~isnan(panelData));
            if ~isempty(panelData)
                yLo = prctile(panelData, 1);
                yHi = prctile(panelData, 99);
                pad = 0.10 * (yHi - yLo + eps);
                ylim(ax, [yLo-pad, yHi+pad]);
            else
                ylim(ax, yLimGlobal);
            end

            % ---- ticks ----
            yl = ylim(ax);
            yticks(ax, linspace(yl(1), yl(2), 5));

            yline(ax, 0, 'k--', 'LineWidth', 0.5, 'HandleVisibility','off');
            xlim(ax, [0 1]);
            xticks(ax, 0:0.2:1);
            set(ax, 'FontSize', fontPanel);
            set(ax, 'LineWidth', 0.8);
        end
    end

    % x-label only on bottom row
    for mIdx = 1:nModelCols
        ax = lastAxPerModel(mIdx);
        if ~isempty(ax) && isgraphics(ax)
            xlabel(ax, 'Normalized time (0--1)', 'Interpreter','latex', 'FontSize', 11);
        end
    end

    % legend (same position as your Big Figure)
    axLeg = axes('Parent',fig,'Units','normalized','Position',[0.14 0.01 0.60 0.06]);
    axis(axLeg,'off'); hold(axLeg,'on');

    hLeg = gobjects(nSubj+1, 1);
    legText = cell(nSubj+1, 1);

    for s = 1:nSubj
        hLeg(s) = plot(axLeg, nan, nan, '-', 'LineWidth', 2.5, 'Color', colSub(s,:));
        legText{s} = sprintf('$\\mathrm{Subject\\ %d}$', s);
    end
    hLeg(nSubj+1) = plot(axLeg, nan, nan, 'k-', 'LineWidth', 3.0);
    legText{nSubj+1} = '$\mathrm{Mean}$';

    legend(axLeg, hLeg, legText, ...
        'Orientation', 'vertical', ...
        'Location', 'northeast', ...
        'Box', 'off', ...
        'Interpreter', 'latex', ...
        'FontSize', 13);

    % export
    outNameSlope = 'BigFigure_VOLonly_SLOPE_TermByModel.pdf';

    set(fig,'Renderer','painters');
    set(fig,'PaperUnits','points');
    set(fig,'PaperSize',[figW figH]);
    set(fig,'PaperPosition',[0 0 figW figH]);
    set(fig,'PaperPositionMode','manual');
    set(fig,'PaperOrientation','portrait');

    print(fig, outNameSlope, '-dpdf', '-painters');
    fprintf('✓ Saved: %s\n', outNameSlope);

end

%% ===== helper: central-difference slope per bin =====
function dy = localSlopeSlidingWin(y, x, win)
% Sliding-window slope per bin:
% For each center bin k, fit y ~ a + b*x using bins [k-h : k+h], h=floor(win/2)
% dy(k) = b. If window contains NaNs, we drop those points; need >=3 points to fit.

y = y(:)'; 
x = x(:)';

K = numel(y);
dy = nan(1, K);

if win < 3
    error('win must be >=3');
end
if mod(win,2) == 0
    error('win must be odd (e.g., 5,7,9)');
end

h = floor(win/2);

for k = 1:K
    lo = max(1, k-h);
    hi = min(K, k+h);

    yy = y(lo:hi);
    xx = x(lo:hi);

    ok = ~isnan(yy) & ~isnan(xx);
    if sum(ok) < 3
        dy(k) = nan;
        continue;
    end

    % linear fit: yy = b0 + b1*xx
    X = [ones(sum(ok),1), xx(ok)'];
    b = X \ yy(ok)';          % b(2) is slope
    dy(k) = b(2);
end
end



%% ===================== LOCAL FUNCTIONS (DO NOT DELETE) =====================

function plot_singleModel_allTerms(SelBest, t_norm, colSub, outPDF)
% Plot ALL betas (terms) for the BEST model in a single grid
% SelBest fields: .beta_sub (nSubj x K x nTerms), .se_sub, .termLabels, .mName

nSubj = size(SelBest.beta_sub, 1);
x = t_norm(:)';
termLabels = SelBest.termLabels;
nTerms = numel(termLabels);

% compute global y-lim
allData = [];
for tt = 1:nTerms
    b = SelBest.beta_sub(:,:,tt);
    e = SelBest.se_sub(:,:,tt);
    allData = [allData; b(:); b(:)+e(:); b(:)-e(:)]; %#ok<AGROW>
end
allData = allData(~isnan(allData));
yMin = min(allData); yMax = max(allData);
pad = 0.06*(yMax-yMin+eps);
yLim = [yMin-pad, yMax+pad];

% layout
nCols = 4;
nRows = ceil(nTerms/nCols);

fig = figure('Color','w','Units','points','Position',[50 50 1200 300*nRows]);

for tt = 1:nTerms
    ax = subplot(nRows, nCols, tt);
    hold(ax,'on'); grid(ax,'on'); box(ax,'off');

    labelStr = termLabels{tt};
    labelStr = strrep(labelStr, 'b0 (Intercept)', '$b_0$ (Intercept)');
    labelStr = strrep(labelStr, 'b_{perf}', '$b_{\mathrm{perf}}$');
    labelStr = strrep(labelStr, 'b_{corr}', '$b_{\mathrm{corr}}$');
    labelStr = strrep(labelStr, 'b_{vol}', '$b_{\mathrm{vol}}$');
    labelStr = strrep(labelStr, 'b_{rt}',  '$b_{\mathrm{rt}}$');
    labelStr = strrep(labelStr, '×', '\times');

    if ~startsWith(labelStr, '$')
        labelStr = ['$' strrep(labelStr, '_', '\_') '$'];
    end

    title(ax, labelStr, 'Interpreter','latex', 'FontSize', 10);

    beta_sub = SelBest.beta_sub(:,:,tt);
    se_sub   = SelBest.se_sub(:,:,tt);

    % subject lines + ribbons
    for s = 1:nSubj
        yv = squeeze(beta_sub(s,:));
        ev = squeeze(se_sub(s,:));
        ok = ~isnan(yv) & ~isnan(ev);

        if sum(ok) >= 2
            xx = x(ok); yy = yv(ok); ee = ev(ok);
            fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(s,:), ...
                'EdgeColor','none', 'FaceAlpha', 0.12, 'HandleVisibility','off');
        end
        plot(ax, x, yv, '-', 'Color', colSub(s,:), 'LineWidth', 0.6, 'HandleVisibility','off');
    end

    % mean ± SEM across subjects
    yMean = mean(beta_sub,1,'omitnan');
    ySEM  = std(beta_sub,0,1,'omitnan') ./ sqrt(nSubj);

    okm = ~isnan(yMean) & ~isnan(ySEM);
    if sum(okm) >= 2
        xx = x(okm); ym = yMean(okm); es = ySEM(okm);
        fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
            'EdgeColor','none', 'FaceAlpha', 0.08, 'HandleVisibility','off');
    end
    plot(ax, x, yMean, 'k-', 'LineWidth', 1.2, 'HandleVisibility','off');

    yline(ax,0,'k--','LineWidth',0.6,'HandleVisibility','off');
    xlim(ax,[0 1]); ylim(ax,yLim);
    xticks(ax,0:0.2:1);

    if tt > nTerms - nCols
        xlabel(ax,'Normalized time (0--1)','Interpreter','latex');
    else
        set(ax,'XTickLabel',[]);
    end
end

% suptitle
if exist('sgtitle','file') == 2
    sgtitle(['Best Model: ' strrep(SelBest.mName,'_','\_') ' - All Betas'], 'FontWeight','bold');
else
    annotation(fig,'textbox',[0 0.965 1 0.03], ...
        'String',['Best Model: ' strrep(SelBest.mName,'_','\_') ' - All Betas'], ...
        'EdgeColor','none','HorizontalAlignment','center','FontWeight','bold');
end

% export
set(fig,'Renderer','painters');
set(fig,'PaperUnits','points');
pos = get(fig,'Position'); figW = pos(3); figH = pos(4);
set(fig,'PaperSize',[figW figH]);
set(fig,'PaperPosition',[0 0 figW figH]);
print(fig, outPDF, '-dpdf', '-painters');
fprintf('✓ Saved: %s\n', outPDF);

end

function dy = localSlopeWindow(y, x, win)
% Compute per-bin local slope using sliding window linear fit.
% y: 1xK or Kx1
% x: 1xK (time)
% win: window length in bins, e.g., 5
%
% Returns dy: 1xK, where dy(k) is slope estimated from bins k..k+win-1.
% Last (win-1) points are NaN (no full window). If you prefer centered
% windows, tell me and I’ll switch.

y = y(:)';      % row
x = x(:)';      % row
K = numel(y);

dy = nan(1, K);
if K < win || win < 2, return; end

for k = 1:(K-win+1)
    idx = k:(k+win-1);
    yy = y(idx);
    xx = x(idx);

    ok = ~isnan(yy) & ~isnan(xx);
    if sum(ok) < 2
        continue;
    end

    p = polyfit(xx(ok), yy(ok), 1);  % linear fit
    dy(k) = p(1);                    % slope
end
end

