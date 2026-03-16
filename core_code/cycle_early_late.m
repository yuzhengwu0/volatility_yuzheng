%% regression_with_rt_earlyLateSplit.m
% PURPOSE:
%   Take each subject's FIRST nEarly trials and LAST nLate trials,
%   run the SAME time-bin logistic regression family (with RT + interactions),
%   do model selection (by BIC/AIC deltas) separately for EARLY vs LATE,
%   then PRINT two big figures:
%     (1) EARLY: top4 models, ALL terms (including interactions)
%     (2) LATE : top4 models, ALL terms (including interactions)
%
% NOTES:
%   - This script computes p_perf_all (RPF) and resVol_time (motion_energy residual volatility)
%     ONCE using all valid trials, then subsets trials for EARLY/LATE fits.
%   - Figures are NOT saved to PDF. They are printed (shown) only.
%
% YOU CAN TUNE:
%   nEarly = 100;   nLate = 100;   and SLOPE_WIN etc if you want later.

clear; clc;

%% ===================== USER SETTINGS =====================
nEarlyCycles = 10;   % 前 n 个循环(cycle/循环)
nLateCycles  = 10;   % 后 m 个循环(cycle/循环)

% 你也可以想要不对称：比如前10个循环，后6个循环
% nEarlyCycles = 10;
% nLateCycles  = 6;

thConf = 0.5;     % binarize confidence

nBins  = 40;      % time bins
minN_pooled = 50; % minimum pooled obs per bin
minN_sub    = 25;  % minimum obs per subject per bin for refit

useSubjDummies = true;  % pooled selection stage only (adds S2,S3)

FORCE_FIXED_MODELS = true;
fixedTopIdx = [2 8 9 10];   % M1, M7, M8, M9




% plotting
DO_PLOT = true;

% subject colors (3 subjects)
colSub = [0 0.4470 0.7410; 1 0 0; 0.9290 0.6940 0.1250];

%% ===================== 0) Toolboxes ======================
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/Palamedes1'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/boundedline-pkg-master'));
RPF_check_toolboxes;

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
Conf = double(confCont >= thConf);

%% ===================== 2) Volatility condition index (RPF) =================
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;

%% ===================== 3) RPF -> predicted performance p_perf_all ==========
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

%% ===================== 4) residual volatility from motion_energy ===========
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

    tmpv = nan(size(y));
    tmpv(mask_b) = resid;
    resVol_mat(:, b) = tmpv;
end

mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_time = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d bins\n', size(resVol_time,1), size(resVol_time,2));

%% ===================== 5) predictors: perf/corr/rt =========================
eps0   = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);
Fp_all = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

Cz_all = Correct - mean(Correct,'omitnan');

rt_eps  = 1e-6;
rt_ref  = log(rt + rt_eps);   % log-RT
RTz_all = (rt_ref - mean(rt_ref,'omitnan')) ./ std(rt_ref,'omitnan');

%% ===================== 6) Build EARLY/LATE indices per subject (by CYCLE) ===
% One cycle = all 6 coherence levels × 2 volatility levels = 12 combinations
% EARLY = first nEarlyCycles cycles (complete cycles)
% LATE  = last  nLateCycles  cycles (complete cycles)
%% ===================== 6) Build EARLY/LATE indices per subject (by CYCLE) ===
% One cycle = all 6 coherence levels × 2 volatility levels = 12 combinations
% EARLY = first nEarlyCycles cycles (complete cycles)
% LATE  = last  nLateCycles  cycles (complete cycles)

idxEarly = false(nTrials,1);
idxLate  = false(nTrials,1);

% Define the levels explicitly (safer, matches your description)
cohLevels = [0 32 64 128 256 512];     % coherence (相干度)
volLevels = [0 256];                  % volatility (波动性)

nCoh = numel(cohLevels);
nVol = numel(volLevels);
nComb = nCoh * nVol;  % should be 12

for iSub = 1:nSubj
    s = subj_list(iSub);

    % subject trial indices in original order
    idxS = find(subjID == s);
    if isempty(idxS), continue; end

    coh_s = coh(idxS);
    vol_s = vol(idxS);

    % Track which combinations have appeared within the current cycle
    seen = false(nCoh, nVol);

    cycleNum = 1;
    cycleStartPos = 1;             % position within idxS
    cycleRanges = [];              % rows: [startPos endPos] in idxS-space

    for p = 1:numel(idxS)
        cval = coh_s(p);
        vval = vol_s(p);

        ic = find(cval == cohLevels, 1, 'first');
        iv = find(vval == volLevels, 1, 'first');

        % If this trial is not one of the expected levels, skip it (or error)
        if isempty(ic) || isempty(iv)
            % warning('Subject %d: unexpected coh/vol at pos %d (coh=%g, vol=%g). Skipping this trial for cycle counting.', ...
            %     s, p, cval, vval);
            continue;
        end

        seen(ic, iv) = true;

        % If we have collected all 12 combos, we complete one cycle here
        if all(seen(:))
            cycleRanges(end+1, :) = [cycleStartPos, p]; %#ok<AGROW>
            cycleNum = cycleNum + 1;

            % reset for next cycle
            seen = false(nCoh, nVol);
            cycleStartPos = p + 1;
        end
    end

    nCompleteCycles = size(cycleRanges, 1);

    % If no complete cycle, nothing to do
    if nCompleteCycles == 0
        continue;
    end

    % Decide how many cycles to take from the start/end
    nE = min(nEarlyCycles, nCompleteCycles);
    nL = min(nLateCycles,  nCompleteCycles);

    % ---- EARLY: cycles 1..nE ----
    for cc = 1:nE
        st = cycleRanges(cc,1);
        en = cycleRanges(cc,2);
        idxEarly(idxS(st:en)) = true;
    end

    % ---- LATE: last nL cycles ----
    for cc = (nCompleteCycles - nL + 1) : nCompleteCycles
        st = cycleRanges(cc,1);
        en = cycleRanges(cc,2);
        idxLate(idxS(st:en)) = true;
    end
end

fprintf('\nEARLY total trials (by cycles): %d\n', sum(idxEarly));
fprintf('LATE  total trials (by cycles): %d\n', sum(idxLate));




%% ===================== 7) Model family (same as your NEW family) ============
[modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels] = build_model_family();

%% ===================== 8) Run pipeline for EARLY and LATE ===================
if DO_PLOT
    SelEarly = run_split_and_plot( ...
        idxEarly, 'EARLY', ...
        Conf, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, colSub, ...
        modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
        useSubjDummies, minN_pooled, minN_sub, ...
        FORCE_FIXED_MODELS, fixedTopIdx);

    SelLate = run_split_and_plot( ...
        idxLate, 'LATE', ...
        Conf, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, colSub, ...
        modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
        useSubjDummies, minN_pooled, minN_sub, ...
        FORCE_FIXED_MODELS, fixedTopIdx);

    % 用 4 列合并图替代原来的 early / late 分开图
    plot_bigfigure_4cols_M8M9_earlyLate(SelEarly, SelLate, t_norm, colSub, []);
end

fprintf('\nDone.\n');

%% ========================================================================
%% ============================ LOCAL FUNCTIONS ============================
%% ========================================================================

function SelOut = run_split_and_plot(idxSplit, splitTag, ...
    Conf, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, colSub, ...
    modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
    useSubjDummies, minN_pooled, minN_sub, ...
    FORCE_FIXED_MODELS, fixedTopIdx)


% ---- pooled model selection (AIC/BIC per bin) ----
[AIC_mat, BIC_mat] = fit_models_pooled( ...
    idxSplit, Conf, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, ...
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
    idxSplit, Conf, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, ...
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
    idxSplit, Conf, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, ...
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
            & ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(subjID) & ~isnan(RTz_all);

        if sum(mask) < minN, continue; end

        y    = Conf(mask);
        P    = Fp_all(mask);
        C    = Cz_all(mask);
        Vraw = Vk(mask);
        R    = RTz_all(mask);
        sID  = subjID(mask);
        if numel(unique(y)) < 2
            continue;
        end


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

        AIC_mat(k,m) = g.ModelCriterion.AIC;
        BIC_mat(k,m) = g.ModelCriterion.BIC;
    end
end

end

function Sel = refit_models_perSubjectPerBin( ...
    idxSplit, Conf, Correct, subjID, Fp_all, Cz_all, RTz_all, resVol_time, t_norm, ...
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
                & ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(RTz_all);

            if sum(mask) < minN_sub, continue; end

            y    = Conf(mask);
            P    = Fp_all(mask);
            C    = Cz_all(mask);
            Vraw = Vk(mask);
            R    = RTz_all(mask);
            if numel(unique(y)) < 2
                continue;
            end


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
                'VariableNames', {'conf','perf','corr','vol','rt', ...
                                  'PxC','PxV','PxR','VxC','CxR','RxV', ...
                                  'PxVxC','PxCxR','PxVxR','VxCxR', ...
                                  'PxVxCxR'});

            % formula
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

function plot_bigfigure_allTerms(Sel, top4Names, t_norm, colSub, splitTag, yLimGlobalForced)

% Big grid:
%   rows = all terms present across the 4 models (union of termLabels)
%   cols = 4 models
% Each panel: subject curves + mean, with subject ribbons using SE.

nModels = numel(Sel);
nSubj   = size(Sel(1).beta_sub,1);
x       = t_norm(:)';

% ---- union of terms across models (keep a stable order: base -> 2-way -> 3-way -> 4-way) ----
allTerms = {};
for m = 1:nModels
    allTerms = [allTerms, Sel(m).termLabels]; %#ok<AGROW>
end
allTerms = unique(allTerms, 'stable');

nRows = numel(allTerms);
nCols = nModels;

% ---- global y-lim across ALL panels (for comparability) ----
allData = [];
for r = 1:nRows
    for m = 1:nModels
        termList = Sel(m).termLabels;
        tt = find(strcmp(termList, allTerms{r}), 1, 'first');
        if isempty(tt), continue; end
        b = Sel(m).beta_sub(:,:,tt);
        e = Sel(m).se_sub(:,:,tt);
        allData = [allData; b(:); b(:)+e(:); b(:)-e(:)]; %#ok<AGROW>
    end
end
allData = allData(~isnan(allData));
if isempty(allData)
    warning('plot_bigfigure_allTerms: allData empty. Skipping plot.');
    return;
end

lo = prctile(allData, 2);
hi = prctile(allData, 98);
pad = 0.08*(hi-lo+eps);
yLimShared = [lo-pad, hi+pad];


% ---- layout (similar to your tile approach, but flexible for many rows) ----
tileSize = 120;
gapX = 18;
gapY = 14;
labelW = 110;

outerL = 40; outerR = 30; outerT = 40; outerB = 65;

figW = outerL + labelW + nCols*tileSize + (nCols-1)*gapX + outerR;
figH = outerT + nRows*tileSize + (nRows-1)*gapY + outerB;

fig = figure('Color','w','Units','points','Position',[60 60 figW figH]);
set(fig, 'Name', sprintf('%s: Top4 models (ALL terms)', splitTag));

pt2nx = @(pt) pt / figW;
pt2ny = @(pt) pt / figH;

L = pt2nx(outerL); R = pt2nx(outerR);
T = pt2ny(outerT); B = pt2ny(outerB);

gapXNorm = pt2nx(gapX);
gapYNorm = pt2ny(gapY);
labelWNorm = pt2nx(labelW);

tileWNorm = pt2nx(tileSize);
tileHNorm = pt2ny(tileSize);

usedW = labelWNorm + nCols*tileWNorm + (nCols-1)*gapXNorm;
usedH = nRows*tileHNorm + (nRows-1)*gapYNorm;

x0   = L + ((1 - L - R) - usedW)/2;
yTop = 1 - T - ((1 - T - B) - usedH)/2;

% latex-safe model names
modelNamesLatex = cell(1,nCols);
for m = 1:nCols
    nm = top4Names{m};
    nm = strrep(nm,'_','\_');
    modelNamesLatex{m} = nm;
end

fontPanel = 9;
lw_sub    = 0.55;
lw_mean   = 1.00;
alphaSub  = 0.12;
alphaMean = 0.08;

lastAxPerModel = gobjects(1,nCols);

for r = 1:nRows
    yPos = yTop - r*tileHNorm - (r-1)*gapYNorm;

    % left label
    axLab = axes('Parent',fig,'Units','normalized','Position',[x0, yPos, labelWNorm, tileHNorm]);
    axis(axLab,'off');
    text(axLab, 0.50, 0.50, term_to_tex_compact(allTerms{r}), ...
        'FontSize', 9, 'FontWeight','bold', ...
        'Rotation', 90, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'Interpreter','tex', ...     % ✅ 用 tex
        'Clipping','on');


    for m = 1:nCols
        xPos = x0 + labelWNorm + (m-1)*(tileWNorm + gapXNorm);
        ax = axes('Parent',fig,'Units','normalized','Position',[xPos, yPos, tileWNorm, tileHNorm]);
        hold(ax,'on'); grid(ax,'on'); box(ax,'off');

        if r == 1
            title(ax, ['$\mathrm{' modelNamesLatex{m} '}$'], ...
                'Interpreter','latex','FontSize',11,'FontWeight','bold');
        end

        termList = Sel(m).termLabels;
        tt = find(strcmp(termList, allTerms{r}), 1, 'first');
        if isempty(tt)
            axis(ax,'off'); set(ax,'Visible','off');
            continue;
        end

        lastAxPerModel(m) = ax;

        beta_sub = Sel(m).beta_sub(:,:,tt);
        se_sub   = Sel(m).se_sub(:,:,tt);

        % subject lines + ribbons
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

        % mean ± SEM
        yMean = mean(beta_sub,1,'omitnan');
        ySEM  = std(beta_sub,0,1,'omitnan') ./ sqrt(nSubj);
        okm = ~isnan(yMean) & ~isnan(ySEM);
        if sum(okm) >= 2
            xx = x(okm); ym = yMean(okm); es = ySEM(okm);
            fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                'EdgeColor','none','FaceAlpha',alphaMean,'HandleVisibility','off');
        end
        plot(ax, x, yMean, 'k-', 'LineWidth', lw_mean, 'HandleVisibility','off');

        yline(ax,0,'k--','LineWidth',0.6,'HandleVisibility','off');
        xlim(ax,[0 1]);
        % 若外部传了统一 y 轴，就强制用它；否则用本函数算出来的 yLimGlobal
        if exist('yLimGlobalForced','var') && ~isempty(yLimGlobalForced)
            ylim(ax, yLimGlobalForced);
        else
            ylim(ax, yLimShared);   % ✅ 用你刚算的 percentile range
        end



        xticks(ax,0:0.2:1);
        set(ax,'FontSize',fontPanel,'LineWidth',0.8);

        if r < nRows
            set(ax,'XTickLabel',[]);
        end
    end
end

% x-label only on bottom row
for m = 1:nCols
    ax = lastAxPerModel(m);
    if ~isempty(ax) && isgraphics(ax)
        xlabel(ax,'Normalized time (0--1)','Interpreter','latex','FontSize',10);
    end
end

% legend
axLeg = axes('Parent',fig,'Units','normalized','Position',[0.14 0.01 0.60 0.06]);
axis(axLeg,'off'); hold(axLeg,'on');

hLeg = gobjects(nSubj+1,1);
legText = cell(nSubj+1,1);
for s = 1:nSubj
    hLeg(s) = plot(axLeg, nan, nan, '-', 'LineWidth', 2.5, 'Color', colSub(s,:));
    legText{s} = sprintf('$\\mathrm{Subject\\ %d}$', s);
end
hLeg(nSubj+1) = plot(axLeg, nan, nan, 'k-', 'LineWidth', 3.0);
legText{nSubj+1} = '$\mathrm{Mean}$';

legend(axLeg, hLeg, legText, ...
    'Orientation','vertical','Location','northeast','Box','off', ...
    'Interpreter','latex','FontSize',12);

% figure title
if exist('sgtitle','file') == 2
    sgtitle(sprintf('%s: Top4 models (ALL terms) | per-subject first/last split', splitTag), 'FontWeight','bold');
else
    annotation(fig,'textbox',[0 0.965 1 0.03], ...
        'String',sprintf('%s: Top4 models (ALL terms)', splitTag), ...
        'EdgeColor','none','HorizontalAlignment','center','FontWeight','bold');
end

end

function s = term_to_latex(term)
% Convert your termLabels strings into latex-ish display.
% input examples: 'b_{perf×vol}', 'b0 (Intercept)', 'b_{rt×vol}', etc.

s = term;

% harmonize intercept
s = strrep(s, 'b0 (Intercept)', '$b_0$ (Intercept)');

% wrap if not already latex-wrapped
if ~startsWith(s,'$')
    s = ['$' s '$'];
end

% fix common characters
s = strrep(s, '×', '\times');
s = strrep(s, '_', '\_'); % safe
% BUT keep proper subscripts like b_{...}: undo \_ inside braces a bit
s = strrep(s, 'b\_{', 'b_{');

% make perf/corr/vol/rt roman (optional)
s = strrep(s, '\mathrm{perf}', 'perf');
s = strrep(s, '\mathrm{corr}', 'corr');
s = strrep(s, '\mathrm{vol}',  'vol');
s = strrep(s, '\mathrm{rt}',   'rt');

end

function yLimShared = compute_shared_ylim(SelA, SelB)
% compute y-lim across two Sel structs (EARLY/LATE), using betas ± SE

allData = [];

for Sel = {SelA, SelB}
    S = Sel{1};
    for m = 1:numel(S)
        b = S(m).beta_sub;
        e = S(m).se_sub;
        allData = [allData; b(:); b(:)+e(:); b(:)-e(:)]; %#ok<AGROW>
    end
end

allData = allData(~isnan(allData));
if isempty(allData)
    yLimShared = [-1 1];
    return;
end

yMin = min(allData);
yMax = max(allData);
pad  = 0.06*(yMax - yMin + eps);
yLimShared = [yMin-pad, yMax+pad];
end

function s = term_to_tex_compact(term)
t = string(term);

if contains(t, "Intercept")
    s = '\beta_0';
    return;
end

t = strrep(t, "b_{", "\beta_{");
t = strrep(t, "b_",  "\beta_");

t = strrep(t, "perf", "P");
t = strrep(t, "corr", "C");
t = strrep(t, "vol",  "V");
t = strrep(t, "rt",   "R");

% 保留 × 字符（tex 更能吃）
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
