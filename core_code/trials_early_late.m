%% trials_early_late.m
% PURPOSE:
%   Compare early vs late trials in the confidence-volatility analysis by
%   running the same time-resolved regression pipeline separately for each split.
%
% MAIN PROCEDURE:
%   1) Split each subject's data into FIRST nEarly trials and LAST nLate trials.
%   2) Compute predicted performance using subject × volatility × coherence
%      mean accuracy (p_perf_all).
%   3) Compute time-resolved residual volatility from motion energy
%      separately for the EARLY and LATE trial sets.
%
% REGRESSION ANALYSIS:
%   Linear regression predicting logit-transformed continuous confidence:
%
%       ConfY ~ perf + corr + vol + rt + interactions
%
%   Two levels of regression are performed separately for EARLY and LATE:
%     (1) pooled regression across all subjects at each time bin
%         (used for AIC/BIC model comparison)
%     (2) per-subject regression at each time bin for selected models
%         (used for plotting beta time courses).
%
% OPTIONAL VISUALIZATION:
%   - Combined big figure:
%       compares EARLY vs LATE coefficient time courses side by side
%       for selected models and terms.
%
% DEPENDENT VARIABLE:
%   Confidence in [0,1], slightly shrunk from boundaries and then
%   logit-transformed.
%
% KEY PREDICTORS:
%   perf : predicted performance
%   corr : correctness
%   vol  : residual volatility from motion energy
%   rt   : log-transformed response time
%
% NOTES:
%   - EARLY and LATE splits are defined within each subject.
%   - Residual volatility is computed separately for EARLY and LATE trials.
%   - Pooled model selection can include subject dummy regressors, but per-subject refits do not.

clear; clc;

%% ===================== USER SETTINGS =====================
nEarly = 300;     % first n trials per subject
nLate  = 300;     % last  m trials per subject

thConf = 0.5;     % binarize confidence

nBins  = 40;      % time bins
minN_pooled = 50; % minimum pooled obs per bin
minN_sub    = 5;  % minimum obs per subject per bin for refit

useSubjDummies = true;  % pooled selection stage only (adds S2,S3)

FORCE_FIXED_MODELS = true;
fixedTopIdx = [9 10];   % M8, M9

% plotting
DO_PLOT = true;

% subject colors (3 subjects)
colSub = [0 0.4470 0.7410; 1 0 0; 0.9290 0.6940 0.1250];

%% ===================== 0) Toolboxes ======================
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/boundedline-pkg-master'));

%% ===================== 1) Load data ======================
data_path = '/Users/wuyuzheng/Documents/MATLAB/projects/volatility/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh_all       = allStruct.rdm1_coh(:);
resp_all      = allStruct.req_resp(:);        % 1/2
correct_all   = allStruct.correct(:);         % 1/0
confCont_all  = allStruct.confidence(:);      % 0-1
vol_all       = allStruct.rdm1_coh_std(:);
subjID_all    = allStruct.group(:);
ME_cell_all   = allStruct.motion_energy;
rt_all        = allStruct.rt(:);

% ---- basic validity ----
valid_basic = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
              ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all);

% ---- hard filter confidence to [0,1] ----
valid_conf = (confCont_all >= 0) & (confCont_all <= 1);

valid = valid_basic & valid_conf;

fprintf('Dropped by conf out-of-range: %d trials (%.2f%% of basic-valid)\n', ...
    sum(valid_basic & ~valid_conf), 100*sum(valid_basic & ~valid_conf)/max(1,sum(valid_basic)));

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
Conf = double(confCont >= thConf); %#ok<NASGU>

conf_adj = nan(size(confCont));
subj_list = unique(subjID);
nSubj = numel(subj_list);

for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = subjID == s;

    y = confCont(idxS);
    N = sum(idxS);
    y2 = (y*(N-1) + 0.5) / N;   % shrink away from 0/1
    conf_adj(idxS) = y2;
end

zConf = log(conf_adj ./ (1 - conf_adj));
ConfY = zConf;

%% ===================== 2) Volatility condition index ======================
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;

%% ===================== 3) Condition mean accuracy ======================
% p_perf_all = subject-specific mean accuracy within each volatility condition

p_perf_all = nan(nTrials,1);

cond_list = unique(cond(~isnan(cond)));
coh_list  = unique(coh(~isnan(coh)));

for iSub = 1:nSubj
    thisSub = subj_list(iSub);

    for c = cond_list(:)'
        for h = coh_list(:)'
            mask = (subjID == thisSub) & (cond == c) & (coh == h);

            if any(mask)
                mean_acc = mean(Correct(mask), 'omitnan');
                p_perf_all(mask) = mean_acc;
            end
        end
    end
end

fprintf('Finished subject × volatility × coherence mean accuracy. Valid proportion: %.3f\n', ...
    mean(~isnan(p_perf_all)));

%% ===================== 6) Build EARLY/LATE indices per subject ==============
idxEarly = false(nTrials,1);
idxLate  = false(nTrials,1);

for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = find(subjID == s);   % keeps original order in your data
    if isempty(idxS), continue; end

    nS = numel(idxS);

    nE = min(nEarly, nS);
    nL = min(nLate,  nS);

    idxEarly(idxS(1:nE)) = true;
    idxLate(idxS(end-nL+1:end)) = true;
end

fprintf('\nEARLY total trials: %d\n', sum(idxEarly));
fprintf('LATE  total trials: %d\n', sum(idxLate));
%% ===================== 4) residual volatility from motion_energy (EARLY/LATE separately) ===========
winLen = 10;
tol    = 1e-12;
t_norm = linspace(0, 1, nBins);

resVol_time_early = compute_resVol_time_split(motion_energy, subjID, nBins, winLen, tol, idxEarly);
resVol_time_late  = compute_resVol_time_split(motion_energy, subjID, nBins, winLen, tol, idxLate);

fprintf('Residual volatility (EARLY): %d trials x %d bins\n', size(resVol_time_early,1), size(resVol_time_early,2));
fprintf('Residual volatility (LATE) : %d trials x %d bins\n', size(resVol_time_late,1), size(resVol_time_late,2));
%% ===================== 5) predictors: perf/corr/rt =========================
eps0   = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);
Fp_all = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

Cz_all = Correct - mean(Correct,'omitnan');

rt_eps  = 1e-6;
rt_ref  = log(rt + rt_eps);   % log-RT
RTz_all = (rt_ref - mean(rt_ref,'omitnan')) ./ std(rt_ref,'omitnan');



%% ===================== 7) Model family (same as your NEW family) ============
[modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels] = build_model_family();

%% ===================== 8) Run pipeline for EARLY and LATE ===================
if DO_PLOT
    SelEarly = run_split_and_plot( ...
        idxEarly, 'EARLY', ...
        ConfY, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time_early, t_norm, colSub, ...
        modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
        useSubjDummies, minN_pooled, minN_sub, ...
        FORCE_FIXED_MODELS, fixedTopIdx);

    SelLate = run_split_and_plot( ...
        idxLate, 'LATE', ...
        ConfY, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time_late, t_norm, colSub, ...
        modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
        useSubjDummies, minN_pooled, minN_sub, ...
        FORCE_FIXED_MODELS, fixedTopIdx);

    % 统一 y 轴范围：early+late + (M8/M9) 全部一起取更宽
    plot_bigfigure_4cols_M8M9_earlyLate(SelEarly, SelLate, t_norm, colSub, []);

end
fprintf('confCont range: [%.4f, %.4f], mean=%.4f, sd=%.4f\n', ...
    min(confCont), max(confCont), mean(confCont,'omitnan'), std(confCont,'omitnan'));
fprintf('\nDone.\n');

%% ========================================================================
%% ============================ LOCAL FUNCTIONS ============================
%% ========================================================================

function SelOut = run_split_and_plot(idxSplit, splitTag, ...
    ConfY, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, colSub, ...
    modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
    useSubjDummies, minN_pooled, minN_sub, ...
    FORCE_FIXED_MODELS, fixedTopIdx)


% ---- pooled model selection (AIC/BIC per bin) ----
[AIC_mat, BIC_mat] = fit_models_pooled( ...
    idxSplit, ConfY, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, ...
    modelNames, modelSpec, baseLabels, twoWayNames, threeWayNames, fourWayNames, ...
    useSubjDummies, minN_pooled);

% delta summary
minAIC_perBin = min(AIC_mat, [], 2, 'omitnan');
minBIC_perBin = min(BIC_mat, [], 2, 'omitnan');
deltaAIC_mat  = AIC_mat - minAIC_perBin;
deltaBIC_mat  = BIC_mat - minBIC_perBin;

meanDeltaAIC = mean(deltaAIC_mat, 1, 'omitnan');
medDeltaAIC  = median(deltaAIC_mat, 1, 'omitnan');
meanDeltaBIC = mean(deltaBIC_mat, 1, 'omitnan');
medDeltaBIC  = median(deltaBIC_mat, 1, 'omitnan');

deltaTbl = table(modelNames(:), meanDeltaAIC(:), medDeltaAIC(:), meanDeltaBIC(:), medDeltaBIC(:), ...
    'VariableNames', {'Model','Mean_dAIC','Median_dAIC','Mean_dBIC','Median_dBIC'});

disp(['=== ' splitTag ' Delta AIC/BIC summary ===']);
disp(deltaTbl);

% ---- rank models (same composite score style) ----
if FORCE_FIXED_MODELS
    top4Idx   = fixedTopIdx(:);
    top4Names = modelNames(top4Idx);
else
    % 你原来的 model selection（模型选择）逻辑不变
end

% score = mean([meanDeltaAIC(:), medDeltaAIC(:), meanDeltaBIC(:), medDeltaBIC(:)], 2, 'omitnan');
% [~, rankIdx] = sort(score, 'ascend', 'MissingPlacement','last');
% rankIdx = rankIdx(~isnan(score(rankIdx)));
% 
% N_TOP = 4;
% top4Idx = rankIdx(1:min(N_TOP, numel(rankIdx)));
% top4Names = modelNames(top4Idx);
% 
% disp(['=== ' splitTag ' Top4 models (composite score) ===']);
% disp(table(top4Names(:), score(top4Idx), 'VariableNames', {'Model','CompositeScore'}));

% ---- refit per subject per bin for top4 models (no subj dummies here) ----
Sel = refit_models_perSubjectPerBin( ...
    idxSplit, ConfY, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, ...
    modelNames, modelSpec, baseLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
    top4Idx, minN_sub);

% ---- plot big figure: ALL TERMS x TOP4 MODELS ----
% plot_bigfigure_allTerms(Sel, top4Names, t_norm, colSub, splitTag);
SelOut = Sel;
end

function [modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels] = build_model_family()

baseLabels = {'b0 (Intercept)','b_{perf}','b_{corr}','b_{vol}','b_{rt}'};

twoWayNames  = ["PxC","PxV","PxR","VxC","CxR","RxV"];
twoWayLabels = {'b_{perf×corr}','b_{perf×vol}','b_{perf×rt}', ...
                'b_{vol×corr}','b_{corr×rt}','b_{rt×vol}'};

threeWayNames  = ["PxVxC","PxCxR","PxVxR","VxCxR"];
threeWayLabels = {'b_{perf×vol×corr}','b_{perf×corr×rt}', ...
                  'b_{perf×vol×rt}','b_{vol×corr×rt}'};

fourWayNames  = "PxVxCxR";
fourWayLabels = {'b_{perf×vol×corr×rt}'};

modelNames = {};
modelSpec  = struct('use2',{},'use3',{},'use4',{});

% M0
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M0_base';
modelSpec(idx).use2  = false(1,6);
modelSpec(idx).use3  = false(1,4);
modelSpec(idx).use4  = false;

% M1: PxC
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M1_PC';
modelSpec(idx).use2  = false(1,6);
modelSpec(idx).use2(1) = true;
modelSpec(idx).use3  = false(1,4);
modelSpec(idx).use4  = false;

% M2–M6: one 2-way at a time (excluding PxC)
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

% M9: full
idx = numel(modelNames) + 1;
modelNames{idx}      = 'M9_full';
modelSpec(idx).use2  = true(1,6);
modelSpec(idx).use3  = true(1,4);
modelSpec(idx).use4  = true;

end

function [AIC_mat, BIC_mat] = fit_models_pooled( ...
    idxSplit, ConfY, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, ...
    modelNames, modelSpec, baseLabels, twoWayNames, threeWayNames, fourWayNames, ...
    useSubjDummies, minN)

K = size(resVol_time,2);
nModels = numel(modelNames);

AIC_mat = nan(K, nModels);
BIC_mat = nan(K, nModels);

for m = 1:nModels
    fprintf('\n=== [%s] Pooled fit: %s ===\n', 'SPLIT', modelNames{m});

    % coefVarNames for extraction
    coefVarNames = ["(Intercept)","perf","corr","vol","rt"];
    for j = 1:6
        if modelSpec(m).use2(j), coefVarNames(end+1) = twoWayNames(j); end %#ok<AGROW>
    end
    for j = 1:4
        if modelSpec(m).use3(j), coefVarNames(end+1) = threeWayNames(j); end %#ok<AGROW>
    end
    if modelSpec(m).use4, coefVarNames(end+1) = fourWayNames; end %#ok<AGROW>
    if useSubjDummies
        % assumes subjects are {1,2,3}
        coefVarNames(end+1:end+2) = ["S2","S3"];
    end

    for k = 1:K
        Vk = resVol_time(:,k);

        mask = idxSplit ...
            & ~isnan(Vk) & ~isnan(ConfY) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(subjID) & ~isnan(RTz_all);

        if sum(mask) < minN, continue; end

        y    = ConfY(mask);
        P    = Fp_all(mask);
        C    = Cz_all(mask);
        Vraw = Vk(mask);
        R    = RTz_all(mask);
        sID  = subjID(mask);

        sv = std(Vraw);
        if sv < 1e-12, continue; end
        V = (Vraw - mean(Vraw)) ./ sv;

        % interactions
        PxC = P.*C; PxV = P.*V; PxR = P.*R;
        VxC = V.*C; CxR = C.*R; RxV = R.*V;

        PxVxC = P.*V.*C;
        PxCxR = P.*C.*R;
        PxVxR = P.*V.*R;
        VxCxR = V.*C.*R;

        PxVxCxR = P.*V.*C.*R;

        % table
        if useSubjDummies
            S2 = double(sID==2);
            S3 = double(sID==3);
            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, S2,S3, ...
                'VariableNames', {'confY','perf','corr','vol','rt', ...
                                  'PxC','PxV','PxR','VxC','CxR','RxV', ...
                                  'PxVxC','PxCxR','PxVxR','VxCxR', ...
                                  'PxVxCxR','S2','S3'});
        else
            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, ...
                'VariableNames', {'confY','perf','corr','vol','rt', ...
                                  'PxC','PxV','PxR','VxC','CxR','RxV', ...
                                  'PxVxC','PxCxR','PxVxR','VxCxR', ...
                                  'PxVxCxR'});
        end

        % formula
        f = "confY ~ perf + corr + vol + rt";
        for j = 1:6
            if modelSpec(m).use2(j), f = f + " + " + twoWayNames(j); end
        end
        for j = 1:4
            if modelSpec(m).use3(j), f = f + " + " + threeWayNames(j); end
        end
        if modelSpec(m).use4, f = f + " + " + fourWayNames; end
        if useSubjDummies, f = f + " + S2 + S3"; end

        try
            g = fitglm(T, f, 'Distribution','normal', 'Link','identity');
        catch
            continue;
        end

        AIC_mat(k,m) = g.ModelCriterion.AIC;
        BIC_mat(k,m) = g.ModelCriterion.BIC;
    end
end

end

function Sel = refit_models_perSubjectPerBin( ...
    idxSplit, ConfY, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, ...
    modelNames, modelSpec, baseLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
    topIdx, minN_sub)

subj_list = unique(subjID(:))';
nSubj = numel(subj_list);
K = numel(t_norm);

Sel = struct();
for ii = 1:numel(topIdx)
    mIdx = topIdx(ii);
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

    nTerms = numel(termNames);
    beta_sub = nan(nSubj, K, nTerms);
    se_sub   = nan(nSubj, K, nTerms);
    p_sub    = nan(nSubj, K, nTerms);

    fprintf('\n--- Refit (per subject/bin): %s ---\n', mName);

    for iSub = 1:nSubj
        s = subj_list(iSub);

        for k = 1:K
            Vk = resVol_time(:,k);

            mask = idxSplit ...
                & (subjID == s) ...
                & ~isnan(Vk) & ~isnan(ConfY) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(RTz_all);

            if sum(mask) < minN_sub, continue; end

            y    = ConfY(mask);
            P    = Fp_all(mask);
            C    = Cz_all(mask);
            Vraw = Vk(mask);
            R    = RTz_all(mask);

            sv = std(Vraw);
            if sv < 1e-12, continue; end
            V = (Vraw - mean(Vraw)) ./ sv;

            % interactions
            PxC = P.*C; PxV = P.*V; PxR = P.*R;
            VxC = V.*C; CxR = C.*R; RxV = R.*V;

            PxVxC = P.*V.*C;
            PxCxR = P.*C.*R;
            PxVxR = P.*V.*R;
            VxCxR = V.*C.*R;

            PxVxCxR = P.*V.*C.*R;

            T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, ...
                'VariableNames', {'confY','perf','corr','vol','rt', ...
                                  'PxC','PxV','PxR','VxC','CxR','RxV', ...
                                  'PxVxC','PxCxR','PxVxR','VxCxR', ...
                                  'PxVxCxR'});

            % formula
            f = "confY ~ perf + corr + vol + rt";
            for j = 1:6
                if modelSpec(mIdx).use2(j), f = f + " + " + twoWayNames(j); end
            end
            for j = 1:4
                if modelSpec(mIdx).use3(j), f = f + " + " + threeWayNames(j); end
            end
            if modelSpec(mIdx).use4, f = f + " + " + fourWayNames; end

            try
                g = fitglm(T, f, 'Distribution','normal', 'Link','identity');
            catch
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

    Sel(ii).mName      = mName;
    Sel(ii).termLabels = termLabels;
    Sel(ii).termNames  = termNames;
    Sel(ii).beta_sub   = beta_sub;
    Sel(ii).se_sub     = se_sub;
    Sel(ii).p_sub      = p_sub;
end

end


function s = term_to_tex_compact(term)
t = string(term);

% intercept
if contains(t, "Intercept")
    s = '\beta_0';
    return;
end

% b_{...} -> \beta_{...}
t = strrep(t, "b_{", "\beta_{");
t = strrep(t, "b_",  "\beta_");

% shorten names
t = strrep(t, "perf", "P");
t = strrep(t, "corr", "C");
t = strrep(t, "vol",  "V");
t = strrep(t, "rt",   "R");

% keep the multiplication sign as a literal (more stable than \times in older MATLAB)
% (if you want, you can also remove it entirely: replace "×" with "")
t = strrep(t, "×", "×");

s = char(t);
end


function plot_bigfigure_4cols_M8M9_earlyLate(SelEarly, SelLate, t_norm, colSub, outPDF, termList, figTitle)
% 4 columns:
%   1) M8 EARLY, 2) M8 LATE, 3) M9 EARLY, 4) M9 LATE
%
% termList (optional):
%   a cellstr specifying which terms (rows) to plot.
%   If empty/not provided -> plot union of all terms across panels (old behavior).
%
% figTitle (optional): sgtitle string

if nargin < 5 || isempty(outPDF)
    outPDF = 'BigFigure_M8M9_EarlyLate_AllTerms.pdf';
end
if nargin < 6
    termList = [];
end
if nargin < 7
    figTitle = 'EARLY vs LATE | M8 vs M9';
end

% ---- locate M8/M9 inside SelEarly/SelLate ----
nameE = string({SelEarly.mName});
nameL = string({SelLate.mName});

iM8E = find(contains(nameE, "M8"), 1, 'first');
iM9E = find(contains(nameE, "M9"), 1, 'first');
iM8L = find(contains(nameL, "M8"), 1, 'first');
iM9L = find(contains(nameL, "M9"), 1, 'first');

if any(isempty([iM8E iM9E iM8L iM9L]))
    error('Did not find M8/M9 in SelEarly/SelLate. Check mName strings.');
end

Panels(1).Sel = SelEarly(iM8E); Panels(1).title = 'M8 (2+3) EARLY';
Panels(2).Sel = SelLate(iM8L);  Panels(2).title = 'M8 (2+3) LATE';
Panels(3).Sel = SelEarly(iM9E); Panels(3).title = 'M9 (2+3+4) EARLY';
Panels(4).Sel = SelLate(iM9L);  Panels(4).title = 'M9 (2+3+4) LATE';

nCols = 4;
nSubj = size(Panels(1).Sel.beta_sub,1);
x     = t_norm(:)';

% ---- union of terms across all 4 panels ----
allTerms = {};
for c = 1:nCols
    allTerms = [allTerms, Panels(c).Sel.termLabels]; %#ok<AGROW>
end
allTerms = unique(allTerms, 'stable');

% ---- if user provides termList, restrict ----
if ~isempty(termList)
    % keep stable order based on the global union (so rows consistent across groups)
    keep = ismember(allTerms, termList);
    allTerms = allTerms(keep);
end

% ---- drop rows that are missing in ALL 4 panels (keep partial rows) ----
keep2 = false(size(allTerms));
for r = 1:numel(allTerms)
    term = allTerms{r};
    hasAny = false;
    for c = 1:nCols
        if any(strcmp(Panels(c).Sel.termLabels, term))
            hasAny = true; break;
        end
    end
    keep2(r) = hasAny;
end
allTerms = allTerms(keep2);

nRows = numel(allTerms);
if nRows == 0
    warning('No terms to plot for this request. Skipping: %s', outPDF);
    return;
end

% ---- compute y-lims per (model, term) using EARLY+LATE pooled ----
yLim_M8 = cell(nRows,1);
yLim_M9 = cell(nRows,1);
for r = 1:nRows
    term = allTerms{r};
    yLim_M8{r} = local_term_ylim_twoPanels(Panels(1).Sel, Panels(2).Sel, term);
    yLim_M9{r} = local_term_ylim_twoPanels(Panels(3).Sel, Panels(4).Sel, term);
end

% ===================== FIG LAYOUT =====================
tileSize = 130;   % 想更大就调这里
gapX     = 20;
gapY     = 14;
labelW   = 90;

outerL = 40;
outerR = 30;
outerT = 35;
outerB = 100;

figW = outerL + labelW + nCols*tileSize + (nCols-1)*gapX + outerR;
figH = outerT + nRows*tileSize + (nRows-1)*gapY + outerB;

fig = figure('Color','w','Units','points','Position',[60 60 figW figH]);
set(fig, 'Name', 'EARLY/LATE combined: M8 vs M9');

pt2nx = @(pt) pt / figW;
pt2ny = @(pt) pt / figH;

L = pt2nx(outerL);
R = pt2nx(outerR);
T = pt2ny(outerT);
B = pt2ny(outerB);

gapXNorm   = pt2nx(gapX);
gapYNorm   = pt2ny(gapY);
labelWNorm = pt2nx(labelW);
tileWNorm  = pt2nx(tileSize);
tileHNorm  = pt2ny(tileSize);

usedW = labelWNorm + nCols*tileWNorm + (nCols-1)*gapXNorm;
usedH = nRows*tileHNorm + (nRows-1)*gapYNorm;

x0   = L + ((1 - L - R) - usedW)/2;
yTop = 1 - T - ((1 - T - B) - usedH)/2;

% ---- style ----
fontPanel = 9;
lw_sub    = 0.55;
lw_mean   = 1.00;
alphaSub  = 0.12;
alphaMean = 0.08;

lastAxPerCol = gobjects(1,nCols);

% ===================== DRAW =====================
for r = 1:nRows
    yPos = yTop - r*tileHNorm - (r-1)*gapYNorm;

    % left label
    axLab = axes('Parent',fig,'Units','normalized','Position',[x0, yPos, labelWNorm, tileHNorm]);
    axis(axLab,'off');
    text(axLab, 0.50, 0.50, term_to_tex_compact(allTerms{r}), ...
        'FontSize', 10, 'FontWeight','bold', ...
        'Rotation', 90, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'Interpreter','tex', ...
        'Clipping','on');

    for c = 1:nCols
        % ===== SPECIAL CASE: use the empty M8 tiles in the 4-way row for legend =====
        isLastRow = (r == nRows);
        termIs4way = contains(string(allTerms{r}), 'b_{perf×vol×corr×rt}'); % 你的四阶项label
        m8_has = any(strcmp(Panels(1).Sel.termLabels, allTerms{r})) || any(strcmp(Panels(2).Sel.termLabels, allTerms{r}));
        m9_has = any(strcmp(Panels(3).Sel.termLabels, allTerms{r})) || any(strcmp(Panels(4).Sel.termLabels, allTerms{r}));

        if isLastRow && termIs4way && ~m8_has && m9_has
            % 用 col1+col2 的 tile 区域做 legend（占两格宽）
            xPos1 = x0 + labelWNorm;                       % col1 起点
            yPos1 = yPos;                                  % 当前行 y
            legW  = 2*tileWNorm + gapXNorm;                % 两格宽 + 中间gap
            legH  = tileHNorm;

            axLeg = axes('Parent',fig,'Units','normalized','Position',[xPos1, yPos1, legW, legH]);
            axis(axLeg,'off'); hold(axLeg,'on');

            hLeg = gobjects(nSubj+1,1);
            legText = cell(nSubj+1,1);
            for s = 1:nSubj
                hLeg(s) = plot(axLeg, nan, nan, '-', 'LineWidth', 2.5, 'Color', colSub(s,:));
                legText{s} = sprintf('Subject %d', s);
            end
            hLeg(nSubj+1) = plot(axLeg, nan, nan, 'k-', 'LineWidth', 3.0);
            legText{nSubj+1} = 'Mean';

            lgd = legend(axLeg, hLeg, legText, 'Orientation','vertical', 'Location','northwest');
            lgd.Box = 'off';
            lgd.FontSize = 11;

            % 然后：跳过 c=1,2 的画图（保留空白/legend），从 c=3 开始正常画
            if c <= 2
                continue;
            end
        end
        xPos = x0 + labelWNorm + (c-1)*(tileWNorm + gapXNorm);
        ax = axes('Parent',fig,'Units','normalized','Position',[xPos, yPos, tileWNorm, tileHNorm]);
        hold(ax,'on'); grid(ax,'on'); box(ax,'off');

        if r == 1
            ttl = strrep(Panels(c).title,'_','\_');
            title(ax, ttl, 'Interpreter','tex','FontSize',11,'FontWeight','bold');
        end

        xlim(ax,[0 1]);
        xticks(ax,0:0.2:1);
        set(ax,'FontSize',fontPanel,'LineWidth',0.8);
        yline(ax,0,'k--','LineWidth',0.6,'HandleVisibility','off');

        if r < nRows
            set(ax,'XTickLabel',[]);
        end
        lastAxPerCol(c) = ax;

        termListHere = Panels(c).Sel.termLabels;
        tt = find(strcmp(termListHere, allTerms{r}), 1, 'first');
        if isempty(tt)
            % should be rare because we filtered all-missing rows,
            % but can still happen per-panel
            text(ax, 0.5, 0.5, '—', 'Units','normalized', ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontSize', 16, 'Color', [0.35 0.35 0.35]);
            if c <= 2, ylim(ax, yLim_M8{r}); else, ylim(ax, yLim_M9{r}); end
            continue;
        end

        beta_sub = Panels(c).Sel.beta_sub(:,:,tt);
        se_sub   = Panels(c).Sel.se_sub(:,:,tt);

        for s = 1:nSubj
            yv = squeeze(beta_sub(s,:));
            ev = squeeze(se_sub(s,:));
            ok = ~isnan(yv) & ~isnan(ev);
            if sum(ok) >= 2
                xx = x(ok); yy = yv(ok); ee = ev(ok);
                fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(s,:), ...
                    'EdgeColor','none','FaceAlpha',alphaSub,'HandleVisibility','off');
            end
            plot(ax, x, yv, '-', 'Color', colSub(s,:), 'LineWidth', lw_sub, 'HandleVisibility','off');
        end

        yMean = mean(beta_sub,1,'omitnan');
        ySEM  = std(beta_sub,0,1,'omitnan') ./ sqrt(nSubj);
        okm = ~isnan(yMean) & ~isnan(ySEM);
        if sum(okm) >= 2
            xx = x(okm); ym = yMean(okm); es = ySEM(okm);
            fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                'EdgeColor','none','FaceAlpha',alphaMean,'HandleVisibility','off');
        end
        plot(ax, x, yMean, 'k-', 'LineWidth', lw_mean, 'HandleVisibility','off');

        if c <= 2
            ylim(ax, yLim_M8{r});
        else
            ylim(ax, yLim_M9{r});
        end
    end
end

for c = 1:nCols
    ax = lastAxPerCol(c);
    if ~isempty(ax) && isgraphics(ax)
        xlabel(ax,'Normalized time (0--1)','Interpreter','latex','FontSize',10);
    end
end


% export
set(fig,'Renderer','painters');
set(fig,'PaperUnits','points');
set(fig,'PaperSize',[figW figH]);
set(fig,'PaperPosition',[0 0 figW figH]);
set(fig,'PaperPositionMode','manual');
set(fig,'PaperOrientation','portrait');

print(fig, outPDF, '-dpdf', '-painters');
fprintf('✓ Saved: %s\n', outPDF);

end

% -------- helper: compute y-lim for one term using two panels pooled (beta ± se) --------
function yLim = local_term_ylim_twoPanels(SelA, SelB, termLabel)
% termLabel is from allTerms (matches Sel.termLabels entries)

vals = [];

for S = {SelA, SelB}
    Sel = S{1};
    tt = find(strcmp(Sel.termLabels, termLabel), 1, 'first');
    if isempty(tt), continue; end

    b = Sel.beta_sub(:,:,tt);
    e = Sel.se_sub(:,:,tt);
    vals = [vals; b(:); b(:)+e(:); b(:)-e(:)]; %#ok<AGROW>
end

vals = vals(~isnan(vals));
if isempty(vals)
    yLim = [-1 1];
    return;
end

% percentile trim to avoid one crazy spike blowing up the panel
lo = prctile(vals, 2);
hi = prctile(vals, 98);

if abs(hi-lo) < 1e-6
    lo = lo - 1;
    hi = hi + 1;
end

pad = 0.10 * (hi - lo + eps);
yLim = [lo-pad, hi+pad];
end

function plot_M8M9_earlyLate_splitByFactor(SelEarly, SelLate, t_norm, colSub, prefix)
% Make 4 PDFs:
%   - volatility terms
%   - perf terms
%   - corr terms
%   - rt terms
%
% Rule: a term appears in any group that matches its string (so interactions duplicate across groups).

if nargin < 5 || isempty(prefix)
    prefix = 'BigFigure_M8M9_EarlyLate';
end

% collect union terms
nameE = string({SelEarly.mName});
nameL = string({SelLate.mName});
iM8E = find(contains(nameE,"M8"),1,'first');
iM9E = find(contains(nameE,"M9"),1,'first');
iM8L = find(contains(nameL,"M8"),1,'first');
iM9L = find(contains(nameL,"M9"),1,'first');
if any(isempty([iM8E iM9E iM8L iM9L]))
    error('Missing M8/M9 in SelEarly/SelLate');
end

Panels = {SelEarly(iM8E), SelLate(iM8L), SelEarly(iM9E), SelLate(iM9L)};
allTerms = {};
for i=1:4
    allTerms = [allTerms, Panels{i}.termLabels]; %#ok<AGROW>
end
allTerms = unique(allTerms,'stable');

% build 4 groups
termVol  = allTerms(cellfun(@(t) local_hasFactor(t,'vol'),  allTerms));
termPerf = allTerms(cellfun(@(t) local_hasFactor(t,'perf'), allTerms));
termCorr = allTerms(cellfun(@(t) local_hasFactor(t,'corr'), allTerms));
termRT   = allTerms(cellfun(@(t) local_hasFactor(t,'rt'),   allTerms));

% export each
plot_bigfigure_4cols_M8M9_earlyLate( ...
    SelEarly, SelLate, t_norm, colSub, ...
    sprintf('%s_VOL.pdf', prefix), termVol, ...
    'EARLY vs LATE | M8 vs M9 | Terms containing VOL');

plot_bigfigure_4cols_M8M9_earlyLate( ...
    SelEarly, SelLate, t_norm, colSub, ...
    sprintf('%s_PERF.pdf', prefix), termPerf, ...
    'EARLY vs LATE | M8 vs M9 | Terms containing PERF');

plot_bigfigure_4cols_M8M9_earlyLate( ...
    SelEarly, SelLate, t_norm, colSub, ...
    sprintf('%s_CORR.pdf', prefix), termCorr, ...
    'EARLY vs LATE | M8 vs M9 | Terms containing CORR');

plot_bigfigure_4cols_M8M9_earlyLate( ...
    SelEarly, SelLate, t_norm, colSub, ...
    sprintf('%s_RT.pdf', prefix), termRT, ...
    'EARLY vs LATE | M8 vs M9 | Terms containing RT');

end

function tf = local_hasFactor(termLabel, key)
% robust string test for membership
% termLabel examples: 'b_{perf×vol}' or 'b_{perf×vol×corr×rt}' etc.
t = lower(string(termLabel));

% drop intercept
if contains(t,"intercept")
    tf = false;
    return;
end

% normalize
t = erase(t, {'{','}','_','\',' ', '$'});

% match keys
tf = contains(t, lower(key));
end

function resVol_time = compute_resVol_time_split(motion_energy, subjID, nBins, winLen, tol, idxUse)
% Compute residual volatility using only the trials in idxUse
% Other trials remain NaN

nTrials = numel(motion_energy);
subj_list = unique(subjID);
nSubj = numel(subj_list);

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

    mask_b = idxUse & ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 3
        continue;
    end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    resid = y_use - Xb * beta;

    tmpv = nan(size(y));
    tmpv(mask_b) = resid;
    resVol_mat(:, b) = tmpv;
end

% per-subject z-score within the chosen split only
resVol_time = nan(size(resVol_mat));

for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = (subjID == s) & idxUse;

    vals = resVol_mat(idxS, :);
    mu_s = mean(vals(:), 'omitnan');
    sd_s = std(vals(:),  'omitnan');

    if isnan(sd_s) || sd_s < 1e-12
        continue;
    end

    resVol_time(idxS, :) = (vals - mu_s) ./ sd_s;
end
end