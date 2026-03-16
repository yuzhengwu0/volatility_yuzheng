%% rank_sweep_withinSub_trainTest_generate.m
% =========================================================================
% PURPOSE (目的)
%   Within-subject Train/Test split (每个被试内部一半训练一半测试)：
%
%   (A) Build predictors & outcome:
%       - P = predicted performance (RPF)
%       - V = residual volatility from motion-energy (time-resolved, K bins)
%       - R = RT (z-scored)
%       - C = correctness (centered)
%       - y = zConf = logit(boundary-adjusted confidence)
%
%   (B) Time-resolved regressions (per subject, per time-bin):
%       MAIN: zConf ~ 1 + V + P + R + C    -> 5 betas across time
%       INT : + interactions (2/3/4-way)   -> nInt betas across time
%
%   (C) For rank r = 1,2,3,4:
%       - Learn basis W (phi) from TRAIN joint betas (main+int)
%       - Infer each subject constants A using TRAIN MAIN only
%       - Generate FULL joint betas (main+int) from A and W
%
%   (D) Plots (same subject same color):
%       1) TRAIN vs TEST (actual betas) : TRAIN=solid, TEST=dashed
%       2) TEST vs GENERATED            : TEST=solid, GEN=dashed
%       (both MAIN and INT)
%
%   (E) Print constants A for each rank
%
% NOTES (注意)
%   - This script is long but fully self-contained.
%   - Local functions MUST be placed at the end of file (MATLAB rule).
%   - Avoid variable name "hold" because it conflicts with hold on/off.
% =========================================================================

clear; clc; close all;
clear hold; % IMPORTANT: avoid conflict if "hold" exists as a variable
rng('shuffle');


%% =========================
% USER OPTIONS (可改参数)
% =========================
% smoothing betas across time (平滑 betas)
DO_SMOOTH_BETAS = true;
SMOOTH_WIN      = 3;          % 3 or 5 usually

% interaction order (交互阶数): 2 -> 6 terms, 3 -> 10 terms, 4 -> 11 terms
MAX_INT_ORDER = 4;

% regression minimum samples per (subject,timebin)
minN = 30;
sv_tol = 1e-12;

% time bins
K = 40;
t_norm = linspace(0,1,K);

% rank sweep
rList = [1 2 3];

% infer A using MAIN only (recommended to prevent leakage)
%INFER_A_FROM_MAIN_ONLY = true;

% ===== Infer A options =====
INFER_A_FROM_MAIN_ONLY = false;   % 你现在要带交互项一起推 A，所以设 false
INT_A_WEIGHT = 0.30;              % INT 权重（0.05~0.3 常见，先 0.2）


% when inferring A: use all time bins or only a window
A_INFER_TIME_MODE = "all";        % "all" or "window"
A_WIN_RANGE = [0.20 0.80];        % used if mode="window"

%% =========================
% 0) Toolboxes (你的路径)
% =========================
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/Palamedes1'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/boundedline-pkg-master'));
RPF_check_toolboxes;

%% =========================
% 1) Load data
% =========================
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh_all       = allStruct.rdm1_coh(:);
resp_all      = allStruct.req_resp(:);
correct_all   = allStruct.correct(:);
confCont_all  = allStruct.confidence(:);
vol_all       = allStruct.rdm1_coh_std(:);
subjID_all    = allStruct.group(:);
ME_cell_all   = allStruct.motion_energy;
rt_all        = allStruct.rt(:);

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

subj_list = unique(subjID);
S = numel(subj_list);
fprintf('Subjects S = %d\n', S);

%% =========================
% 2) Map volatility levels -> condition index (RPF)
% =========================
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;

%% =========================
% 3) RPF -> predicted performance P (trial-level)
% =========================
p_perf_all = nan(nTrials, 1);

for iSub = 1:S
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

%% =========================
% 4) motion_energy -> residual volatility resVol_time (trial x K)
% =========================
winLen = 10;
tol = 1e-12;

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

MEAN_norm = nan(nTrials, K);
STD_norm  = nan(nTrials, K);

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
for b = 1:K
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

%% =========================
% 5) Build predictors + outcome zConf
% =========================
% Perf z-score (global)
eps0   = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);
Pz_all = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

% Corr centered
Cz_all = Correct - mean(Correct,'omitnan');

% RT z-score (log RT)
rt_ref  = log(rt + 1e-6);
RTz_all = (rt_ref - mean(rt_ref,'omitnan')) ./ std(rt_ref,'omitnan');

% Confidence boundary adjust per subject then logit
conf_adj = nan(size(confCont));
for iSub = 1:S
    s = subj_list(iSub);
    idxS = (subjID == s);
    y = min(max(confCont(idxS),0),1);
    N = sum(idxS);
    y2 = (y*(N-1) + 0.5) / N;   % Smithson & Verkuilen
    conf_adj(idxS) = y2;
end
zConf_all = log(conf_adj ./ (1 - conf_adj));

%% =========================
% 6) Within-subject Train/Test split (每个被试一半一半)
% =========================
trainMask = false(nTrials,1);
testMask  = false(nTrials,1);

for iSub = 1:S
    s = subj_list(iSub);
    idxS = find(subjID == s);

    % keep only trials with valid y and predictors (basic)
    idxS = idxS(~isnan(zConf_all(idxS)) & ~isnan(Pz_all(idxS)) & ...
                ~isnan(Cz_all(idxS)) & ~isnan(RTz_all(idxS)));

    nS = numel(idxS);
    if nS < 40
        warning('Subject %d has few valid trials (%d).', s, nS);
    end

    rp = randperm(nS);
    nTrain = floor(0.50*nS);
    if nTrain < 1, nTrain = 1; end
    trainMask(idxS(rp(1:nTrain))) = true;
    testMask(idxS(rp(nTrain+1:end))) = true;
end

fprintf('Train trials: %d | Test trials: %d\n', sum(trainMask), sum(testMask));
%% =========================
% 6.5) Unified V normalization (ALL trials)
% =========================
muV_all = nan(S, K);
sdV_all = nan(S, K);

for iSub = 1:S
    s = subj_list(iSub);

    for t = 1:K
        Vk = resVol_time(:,t);

        mask_all = (subjID == s) & ~isnan(Vk) & ...
                   ~isnan(zConf_all) & ~isnan(Pz_all) & ...
                   ~isnan(Cz_all) & ~isnan(RTz_all);

        if sum(mask_all) >= 3
            Vraw = Vk(mask_all);
            muV_all(iSub,t) = mean(Vraw);
            sdV_all(iSub,t) = std(Vraw);
        end
    end
end

Nvol_train = nan(S, K);
Nvol_test  = nan(S, K);
Nvol_all   = nan(S, K);

%% =========================
% 7) Fit regressions on TRAIN and on TEST (time-resolved)
% =========================
mainNames = {'Intercept','vol','perf','rt','corr'};
nMain = numel(mainNames);

[intNames, nInt] = getInteractionNames(MAX_INT_ORDER);
fprintf('Interaction order up to %d -> nInt=%d\n', MAX_INT_ORDER, nInt);

% betas: S x K x ...
BetaMainTrain = nan(S, K, nMain);
BetaMainTest  = nan(S, K, nMain);
BetaIntTrain  = nan(S, K, nInt);
BetaIntTest   = nan(S, K, nInt);

% store V normalization separately for TRAIN and TEST
muV_tr = nan(S,K); sdV_tr = nan(S,K);
muV_te = nan(S,K); sdV_te = nan(S,K);

for iSub = 1:S
    s = subj_list(iSub);

    for t = 1:K
        Vk = resVol_time(:,t);

        % ---------- TRAIN ----------
        mask_tr = trainMask & (subjID==s) & ~isnan(Vk) & ~isnan(zConf_all) & ...
                  ~isnan(Pz_all) & ~isnan(Cz_all) & ~isnan(RTz_all);
        Nvol_train(iSub, t) = sum(mask_tr);


        if sum(mask_tr) >= minN
            y = zConf_all(mask_tr);
            P = Pz_all(mask_tr);
            C = Cz_all(mask_tr);
            R = RTz_all(mask_tr);

            Vraw = Vk(mask_tr);

            if sdV_all(iSub,t) > sv_tol
                Vz = (Vraw - muV_all(iSub,t)) ./ sdV_all(iSub,t);

                Xmain = [ones(sum(mask_tr),1), Vz, P, R, C];
                bmain = Xmain \ y;
                BetaMainTrain(iSub,t,:) = bmain(:);

                Xint  = buildInteractionsRow(P, Vz, R, C, MAX_INT_ORDER);
                Xfull = [Xmain, Xint];
                bfull = Xfull \ y;
                BetaIntTrain(iSub,t,:) = bfull(6:(5+nInt));
            end

        end

        % ---------- TEST ----------
        mask_te = testMask & (subjID==s) & ~isnan(Vk) & ~isnan(zConf_all) & ...
                  ~isnan(Pz_all) & ~isnan(Cz_all) & ~isnan(RTz_all);
        
        Nvol_test(iSub, t) = sum(mask_te);

        if sum(mask_te) >= minN
            y = zConf_all(mask_te);
            P = Pz_all(mask_te);
            C = Cz_all(mask_te);
            R = RTz_all(mask_te);

            Vraw = Vk(mask_te);
            %muV_te(iSub,t) = mean(Vraw);
            %sdV_te(iSub,t) = std(Vraw);
            if sdV_all(iSub,t) > sv_tol
                Vz = (Vraw - muV_all(iSub,t)) ./ sdV_all(iSub,t);

                Xmain = [ones(sum(mask_te),1), Vz, P, R, C];
                bmain = Xmain \ y;
                BetaMainTest(iSub,t,:) = bmain(:);

                Xint = buildInteractionsRow(P, Vz, R, C, MAX_INT_ORDER);
                Xfull = [Xmain, Xint];
                bfull = Xfull \ y;

                BetaIntTest(iSub,t,:) = bfull(6:(5+nInt));
            end
        end
    end
end

% smoothing across time (optional)
BetaMainTrainSm = BetaMainTrain;
BetaMainTestSm  = BetaMainTest;
BetaIntTrainSm  = BetaIntTrain;
BetaIntTestSm   = BetaIntTest;

if DO_SMOOTH_BETAS
    for iSub = 1:S
        for c = 1:nMain
            BetaMainTrainSm(iSub,:,c) = movmean(squeeze(BetaMainTrain(iSub,:,c)), SMOOTH_WIN, 'omitnan');
            BetaMainTestSm(iSub,:,c)  = movmean(squeeze(BetaMainTest(iSub,:,c)),  SMOOTH_WIN, 'omitnan');
        end
        for j = 1:nInt
            BetaIntTrainSm(iSub,:,j) = movmean(squeeze(BetaIntTrain(iSub,:,j)), SMOOTH_WIN, 'omitnan');
            BetaIntTestSm(iSub,:,j)  = movmean(squeeze(BetaIntTest(iSub,:,j)),  SMOOTH_WIN, 'omitnan');
        end
    end
end

fprintf('\nDone regression. Now rank sweep...\n');

%% =========================
% 7.5) Fit regressions on ALL trials (time-resolved)  <-- 新增
% =========================
BetaMainAll = nan(S, K, nMain);
BetaIntAll  = nan(S, K, nInt);

muV_allS = nan(S,K); sdV_allS = nan(S,K);

for iSub = 1:S
    s = subj_list(iSub);

    for t = 1:K
        Vk = resVol_time(:,t);

        mask_all = (subjID==s) & ~isnan(Vk) & ~isnan(zConf_all) & ...
                   ~isnan(Pz_all) & ~isnan(Cz_all) & ~isnan(RTz_all);
        Nvol_all(iSub, t) = sum(mask_all);
        
        if sum(mask_all) >= minN
            y = zConf_all(mask_all);
            P = Pz_all(mask_all);
            C = Cz_all(mask_all);
            R = RTz_all(mask_all);

            Vraw = Vk(mask_all);
            muV_allS(iSub,t) = mean(Vraw);
            sdV_allS(iSub,t) = std(Vraw);

            if sdV_allS(iSub,t) > sv_tol
                Vz = (Vraw - muV_allS(iSub,t)) ./ sdV_allS(iSub,t);

                % main
                Xmain = [ones(sum(mask_all),1), Vz, P, R, C];
                bmain = Xmain \ y;
                BetaMainAll(iSub,t,:) = bmain(:);

                % interactions
                Xint  = buildInteractionsRow(P, Vz, R, C, MAX_INT_ORDER);
                Xfull = [Xmain, Xint];
                bfull = Xfull \ y;
                BetaIntAll(iSub,t,:) = bfull(6:(5+nInt));
            end
        end
    end
end

fprintf('\n=== Effective volatility trial counts per subject ===\n');

for iSub = 1:S
    fprintf('Subject %d:\n', subj_list(iSub));

    fprintf('  TRAIN: min=%3d | mean=%5.1f | max=%3d\n', ...
        min(Nvol_train(iSub,:)), mean(Nvol_train(iSub,:)), max(Nvol_train(iSub,:)));

    fprintf('  TEST : min=%3d | mean=%5.1f | max=%3d\n', ...
        min(Nvol_test(iSub,:)),  mean(Nvol_test(iSub,:)),  max(Nvol_test(iSub,:)));

    fprintf('  ALL  : min=%3d | mean=%5.1f | max=%3d\n', ...
        min(Nvol_all(iSub,:)),   mean(Nvol_all(iSub,:)),   max(Nvol_all(iSub,:)));
end


% --- smoothing ALL (match your others) ---
BetaMainAllSm = BetaMainAll;
BetaIntAllSm  = BetaIntAll;

if DO_SMOOTH_BETAS
    for iSub = 1:S
        for c = 1:nMain
            BetaMainAllSm(iSub,:,c) = movmean(squeeze(BetaMainAll(iSub,:,c)), SMOOTH_WIN, 'omitnan');
        end
        for j = 1:nInt
            BetaIntAllSm(iSub,:,j) = movmean(squeeze(BetaIntAll(iSub,:,j)), SMOOTH_WIN, 'omitnan');
        end
    end
end

%% =========================
% 8) Plot: TRAIN vs TEST actual betas (让你先看：beta曲线在train/test一致吗)
% =========================
co = lines(S); % subject colors

plot_main_compare(BetaMainTrainSm, BetaMainTestSm, t_norm, co, ...
    'MAIN betas: TRAIN(solid) vs TEST(dashed)');

plot_int_compare(BetaIntTrainSm, BetaIntTestSm, intNames, t_norm, co, ...
    'INT betas: TRAIN(solid) vs TEST(dashed)');

%% =========================
% 9) Rank sweep: r = 1,2,3,4
%    - Learn W from TRAIN joint betas
%    - Infer A from TRAIN (MAIN-only or JOINT-weighted)
%    - Reconstruct betas (GENERATED)
%    - Compare TRAIN vs GEN  (in-sample reconstruction)
%    - Compare TEST  vs GEN  (generalization)
% =========================
for ir = 1:numel(rList)
    r = rList(ir);
    fprintf('\n==================== RANK r=%d ====================\n', r);

    % ------------------------------------------------------------
    % (1) Learn basis W from TRAIN joint betas (MAIN + INT)
    % ------------------------------------------------------------
    [M_train, info] = packJointBetas(BetaMainTrainSm, BetaIntTrainSm, true); % include intercept
    [W, featMu] = learn_basis_from_train(M_train, r);

    % ------------------------------------------------------------
    % (2) Infer A per subject from TRAIN (choose MAIN-only or JOINT)
    % ------------------------------------------------------------
    if INFER_A_FROM_MAIN_ONLY
        idx_use = get_main_indices_for_A(K, A_INFER_TIME_MODE, A_WIN_RANGE, t_norm);
        w_use   = ones(numel(idx_use),1); % MAIN-only weights
        A = infer_A_weighted_from_joint(BetaMainTrainSm, BetaIntTrainSm, W, featMu, idx_use, w_use, r);
        inferTag = sprintf('TRAIN MAIN only | A_INFER_TIME_MODE=%s', A_INFER_TIME_MODE);
    else
        idx_use = get_joint_indices_for_A(K, nInt, A_INFER_TIME_MODE, A_WIN_RANGE, t_norm);
        w_use   = build_joint_weights(K, nInt, idx_use, INT_A_WEIGHT, A_INFER_TIME_MODE, A_WIN_RANGE, t_norm);
        A = infer_A_weighted_from_joint(BetaMainTrainSm, BetaIntTrainSm, W, featMu, idx_use, w_use, r);
        inferTag = sprintf('TRAIN JOINT weighted (INT=%.2f) | A_INFER_TIME_MODE=%s', INT_A_WEIGHT, A_INFER_TIME_MODE);
    end

    % ------------------------------------------------------------
    % (3) Print constants A
    % ------------------------------------------------------------
    fprintf('Constants A (inferred from %s) for r=%d:\n', inferTag, r);
    for s = 1:S
        fprintf('  Subject %d: ', subj_list(s));
        for k = 1:r
            fprintf('a%d=%+.4f  ', k, A(s,k));
        end
        fprintf('\n');
    end

    % ------------------------------------------------------------
    % (4) Reconstruct joint betas from A and W
    %     m_hat = mu + A*W'
    % ------------------------------------------------------------
    M_hat = (A * W') + featMu; % S x D_joint
    [BetaMainGen, BetaIntGen] = unpackJointBetas(M_hat, info, K, S, true, BetaMainTrainSm);

    % ------------------------------------------------------------
    % (5) TRAIN vs GENERATED (in-sample reconstruction)
    % ------------------------------------------------------------
    plot_main_compare(BetaMainTrainSm, BetaMainGen, t_norm, co, ...
        sprintf('MAIN betas: TRAIN(solid) vs GENERATED(dashed) + ALL(thick dark) | rank r=%d', r), ...
        BetaMainAllSm);

    plot_int_compare(BetaIntTrainSm, BetaIntGen, intNames, t_norm, co, ...
        sprintf('INT betas: TRAIN(solid) vs GENERATED(dashed) + ALL(thick dark) | rank r=%d', r), ...
        BetaIntAllSm);

    fprintf('\n--- Scatter metrics: TRAIN vs GENERATED ---\n');
    print_scatter_metrics(BetaMainTrainSm, BetaMainGen, BetaIntTrainSm, BetaIntGen, mainNames, intNames);

    % ------------------------------------------------------------
    % (6) TEST vs GENERATED (generalization)
    % ------------------------------------------------------------
    plot_main_compare(BetaMainTestSm, BetaMainGen, t_norm, co, ...
        sprintf('MAIN betas: TEST(solid) vs GENERATED(dashed) + ALL(thick dark) | rank r=%d', r), ...
        BetaMainAllSm);

    plot_int_compare(BetaIntTestSm, BetaIntGen, intNames, t_norm, co, ...
        sprintf('INT betas: TEST(solid) vs GENERATED(dashed) + ALL(thick dark) | rank r=%d', r), ...
        BetaIntAllSm);

    fprintf('\n--- Scatter metrics: TEST vs GENERATED ---\n');
    print_scatter_metrics(BetaMainTestSm, BetaMainGen, BetaIntTestSm, BetaIntGen, mainNames, intNames);
    
    print_corr_triplet_metrics(BetaMainTestSm, BetaMainAllSm, BetaMainGen, ...
                               BetaIntTestSm,  BetaIntAllSm,  BetaIntGen, ...
                               mainNames, intNames);

end

fprintf('\nALL DONE.\n');

%% ========================================================================
% ====================== LOCAL FUNCTIONS (必须放在文件末尾) =================
% ========================================================================

function idx_main_use = get_main_indices_for_A(K, mode, winRange, t_norm)
    % MAIN block = 5 blocks of length K
    if mode == "window"
        idxWin = find(t_norm >= winRange(1) & t_norm <= winRange(2));
        if isempty(idxWin), error('A window is empty.'); end
        idx_main_use = [ ...
            idxWin, ...
            (K + idxWin), ...
            (2*K + idxWin), ...
            (3*K + idxWin), ...
            (4*K + idxWin) ...
        ];
    else
        idx_main_use = 1:(5*K);
    end
end

function [W, featMu] = learn_basis_from_train(M_train, r)
    % Learn basis W from TRAIN subjects x features
    M_imp = imputeByFeatureMean(M_train);
    featMu = mean(M_imp, 1);
    M0 = M_imp - featMu;
    [~,~,V] = svd(M0, 'econ');
    W = V(:, 1:r); % D x r
end

function A = infer_A_from_main(BetaMainTrainSm, W, featMu, idx_main_use, r)
    % For each subject s:
    %   Use TRAIN MAIN betas -> solve W_main * a = (m_main - mu_main)
    [S,K,~] = size(BetaMainTrainSm);
    D_main = 5*K;

    A = nan(S, r);

    for s = 1:S
        m_main = pack_main_row(BetaMainTrainSm, s);   % 1 x D_main
        m0 = m_main(idx_main_use) - featMu(idx_main_use);

        W_main = W(idx_main_use, :);                 % Dobs x r

        ok = ~isnan(m0(:)) & all(~isnan(W_main),2);
        if sum(ok) < r
            warning('Subject %d: not enough valid dims to infer A (valid=%d).', s, sum(ok));
            continue;
        end

        a_col = W_main(ok,:) \ m0(ok)'; % r x 1
        A(s,:) = a_col(:)';
    end
end

function row = pack_main_row(BetaMain, iSub)
    % pack MAIN: [b0(1:K), bV(1:K), bP(1:K), bR(1:K), bC(1:K)]
    [~,K,~] = size(BetaMain);
    b0 = squeeze(BetaMain(iSub,:,1));
    bV = squeeze(BetaMain(iSub,:,2));
    bP = squeeze(BetaMain(iSub,:,3));
    bR = squeeze(BetaMain(iSub,:,4));
    bC = squeeze(BetaMain(iSub,:,5));

    row = nan(1, 5*K);
    row(1:K)         = b0(:)';
    row(K+(1:K))     = bV(:)';
    row(2*K+(1:K))   = bP(:)';
    row(3*K+(1:K))   = bR(:)';
    row(4*K+(1:K))   = bC(:)';
end

function plot_main_compare(BetaA, BetaB, t_norm, co, figTitle, BetaAll)
    % BetaA/B: S x K x 5
    % BetaAll (optional): S x K x 5, plotted as thicker darker solid line
    if nargin < 6
        BetaAll = [];
    end

    [S,~,~] = size(BetaA);
    labels = {'b0','bVol','bPerf','bRT','bCorr'};

    figure('Color','w','Position',[180 120 1200 700]);
    sgtitle(figTitle);

    for c = 1:5
        subplot(3,2,c); grid on; box off; hold on;

        for s = 1:S
            yA = squeeze(BetaA(s,:,c));
            yB = squeeze(BetaB(s,:,c));

%             % ---- 1) ALL: 先画在最底层（避免盖住虚线） ----
%             if ~isempty(BetaAll)
%                 yAll = squeeze(BetaAll(s,:,c));
%                 dark = max(min(co(s,:)*0.55, 1), 0);
%                 hAll = plot(t_norm, yAll, '-', 'LineWidth',3.0, 'Color', dark);
%                 uistack(hAll,'bottom'); % 明确压到底（保险）
%             end

            % ---- 2) A: 实线画浅一点、稍细 ----
            colA = co(s,:)*0.35 + 0.65; % 变浅：更接近白色
            hA = plot(t_norm, yA, '-', 'LineWidth',2.0, 'Color', colA);

            % ---- 3) B: 虚线画粗一点，并且放最上层 ----
            hB = plot(t_norm, yB, '--', 'LineWidth',2.4, 'Color', co(s,:));
            uistack(hB,'top');          % 强制虚线在最上

            % （可选）给虚线加一点点 marker，更容易分辨重叠
            % set(hB,'Marker','.', 'MarkerSize',10);
        end

        yline(0,'k--','LineWidth',0.8);
        title(labels{c});
        xlabel('time (0-1)');
        ylabel('\beta');
    end
end


function plot_int_compare(BetaA, BetaB, intNames, t_norm, co, figTitle, BetaAll)
    % BetaA/B: S x K x nInt
    % BetaAll (optional): S x K x nInt, plotted as thicker darker solid line
    if nargin < 7
        BetaAll = [];
    end

    [S,~,nInt] = size(BetaA);
    nC = 3;
    nR = ceil(nInt/nC);

    figure('Color','w','Position',[180 120 1400 820]);
    sgtitle(figTitle);

    for j = 1:nInt
        subplot(nR,nC,j); grid on; box off; hold on;

        for s = 1:S
            yA = squeeze(BetaA(s,:,j));
            yB = squeeze(BetaB(s,:,j));

%             % ALL: 底层
%             if ~isempty(BetaAll)
%                 yAll = squeeze(BetaAll(s,:,j));
%                 dark = max(min(co(s,:)*0.55, 1), 0);
%                 hAll = plot(t_norm, yAll, '-', 'LineWidth',3.0, 'Color', dark);
%                 uistack(hAll,'bottom');
%             end

            % A: 浅色实线
            colA = co(s,:)*0.35 + 0.65;
            hA = plot(t_norm, yA, '-', 'LineWidth',2.0, 'Color', colA);

            % B: 粗虚线 + 最上层
            hB = plot(t_norm, yB, '--', 'LineWidth',2.4, 'Color', co(s,:));
            uistack(hB,'top');

            % 可选 marker
            % set(hB,'Marker','.', 'MarkerSize',10);
        end



        yline(0,'k--','LineWidth',0.8);
        title(['INT ' intNames{j}]);
        xlabel('time (0-1)');
        ylabel('\beta');
    end
end


function print_scatter_metrics(BetaMainTrue, BetaMainHat, BetaIntTrue, BetaIntHat, mainNames, intNames)
    % prints corr/rmse/signMatch for MAIN + INT
    fprintf('\n--- Scatter metrics: TEST vs GENERATED ---\n');

    % MAIN
    for c = 1:5
        a = reshape(BetaMainTrue(:,:,c), [], 1);
        h = reshape(BetaMainHat(:,:,c),  [], 1);
        ok = ~isnan(a) & ~isnan(h);
        if sum(ok) < 3, continue; end
        rr = corr(a(ok), h(ok));
        rmse = sqrt(mean((a(ok)-h(ok)).^2));
        sm = mean(sign(a(ok))==sign(h(ok)));
        fprintf('MAIN %-9s | corr=%+.3f | rmse=%.3f | sign=%.2f | n=%d\n', mainNames{c}, rr, rmse, sm, sum(ok));
    end

    % INT
    nInt = numel(intNames);
    for j = 1:nInt
        a = reshape(BetaIntTrue(:,:,j), [], 1);
        h = reshape(BetaIntHat(:,:,j),  [], 1);
        ok = ~isnan(a) & ~isnan(h);
        if sum(ok) < 3, continue; end
        rr = corr(a(ok), h(ok));
        rmse = sqrt(mean((a(ok)-h(ok)).^2));
        sm = mean(sign(a(ok))==sign(h(ok)));
        fprintf('INT  %-9s | corr=%+.3f | rmse=%.3f | sign=%.2f | n=%d\n', intNames{j}, rr, rmse, sm, sum(ok));
    end
end

function [names, nInt] = getInteractionNames(maxOrder)
    names2 = {'PxV','VxC','RxV','PxC','PxR','CxR'};
    names3 = {'PxVxR','PxVxC','PxRxC','VxRxC'};
    names4 = {'PxVxRxC'};

    if maxOrder == 2
        names = names2;
    elseif maxOrder == 3
        names = [names2, names3];
    elseif maxOrder == 4
        names = [names2, names3, names4];
    else
        error('maxOrder must be 2,3,or 4');
    end
    nInt = numel(names);
end

function Xint = buildInteractionsRow(P, V, R, C, maxOrder)
    x2 = [P.*V, V.*C, R.*V, P.*C, P.*R, C.*R];
    if maxOrder == 2, Xint = x2; return; end

    x3 = [P.*V.*R, P.*V.*C, P.*R.*C, V.*R.*C];
    if maxOrder == 3, Xint = [x2, x3]; return; end

    x4 = [P.*V.*R.*C];
    if maxOrder == 4, Xint = [x2, x3, x4]; return; end

    error('maxOrder must be 2,3,or 4');
end

function M_imp = imputeByFeatureMean(M)
    M_imp = M;
    for d = 1:size(M,2)
        col = M_imp(:,d);
        if all(isnan(col))
            M_imp(:,d) = 0;
        else
            mu = mean(col,'omitnan');
            col(isnan(col)) = mu;
            M_imp(:,d) = col;
        end
    end
end

function [M_joint, info] = packJointBetas(BetaMain, BetaInt, includeIntercept)
    % pack per subject into joint vector: MAIN + INT
    [S,K,~] = size(BetaMain);
    nInt = size(BetaInt,3);

    if includeIntercept
        mainIdx = 1:5;
    else
        mainIdx = 2:5;
    end
    nMainUse = numel(mainIdx);

    D = K*nMainUse + K*nInt;
    M_joint = nan(S, D);

    for s = 1:S
        vec = [];
        % main
        for c = mainIdx
            tmp = squeeze(BetaMain(s,:,c));
            vec = [vec, tmp(:)']; %#ok<AGROW>
        end
        % int
        for j = 1:nInt
            tmp = squeeze(BetaInt(s,:,j));
            vec = [vec, tmp(:)']; %#ok<AGROW>
        end
        M_joint(s,:) = vec;
    end

    info.mainIdx  = mainIdx;
    info.nMainUse = nMainUse;
    info.nInt     = nInt;
end

function [BetaMainHat, BetaIntHat] = unpackJointBetas(Mhat_joint, info, K, S, includeIntercept, BetaMainRef)
    % unpack joint vector back to MAIN and INT betas
    BetaMainHat = nan(S, K, 5);
    BetaIntHat  = nan(S, K, info.nInt);

    if ~includeIntercept
        BetaMainHat(:,:,1) = BetaMainRef(:,:,1);
    end

    ptr = 0;
    for cpos = 1:info.nMainUse
        c = info.mainIdx(cpos);
        cols = ptr + (1:K);
        BetaMainHat(:,:,c) = Mhat_joint(:, cols);
        ptr = ptr + K;
    end

    for j = 1:info.nInt
        cols = ptr + (1:K);
        BetaIntHat(:,:,j) = Mhat_joint(:, cols);
        ptr = ptr + K;
    end
end

function idx_use = get_joint_indices_for_A(K, nInt, mode, winRange, t_norm)
    % JOINT block order (和 packJointBetas 的顺序一致)：
    %   MAIN: 5 blocks of K  -> [b0 bV bP bR bC]
    %   INT : nInt blocks of K
    %
    % return indices into the JOINT vector (length = (5+nInt)*K)

    if mode == "window"
        idxWin = find(t_norm >= winRange(1) & t_norm <= winRange(2));
        if isempty(idxWin), error('A window is empty.'); end

        idx_use = [];
        % MAIN (5 blocks)
        for blk = 0:(5-1)
            idx_use = [idx_use, blk*K + idxWin]; %#ok<AGROW>
        end
        % INT (nInt blocks)
        for j = 0:(nInt-1)
            idx_use = [idx_use, (5+j)*K + idxWin]; %#ok<AGROW>
        end
    else
        idx_use = 1:((5+nInt)*K);
    end
end

function w_use = build_joint_weights(K, nInt, idx_use, intWeight, mode, winRange, t_norm)
    % Build weights for the selected idx_use (a vector, same length as idx_use)
    % MAIN dims weight = 1
    % INT  dims weight = intWeight (e.g., 0.2)

    D = (5 + nInt) * K;
    w_full = ones(D,1);

    % INT blocks start at block index 5
    for j = 1:nInt
        cols = (5*K + (j-1)*K) + (1:K);
        w_full(cols) = intWeight;
    end

    % if using window, idx_use already selects subset; return corresponding weights
    w_use = w_full(idx_use);
end

function A = infer_A_weighted_from_joint(BetaMainTrainSm, BetaIntTrainSm, W, featMu, idx_use, w_use, r)
    % Weighted inference of A using selected dims from JOINT vector
    %
    % Solve:  minimize || sqrt(w) .* (W_use * a - (m_use - mu_use)) ||^2
    % => (diag(sqrt(w)) * W_use) a = diag(sqrt(w)) * (m_use - mu_use)

    [S, K, ~] = size(BetaMainTrainSm);
    nInt = size(BetaIntTrainSm, 3);

    % pack JOINT per subject (same order as packJointBetas)
    % We'll just pack on the fly
    A = nan(S, r);

    sw = sqrt(w_use(:));  % L x 1
    W_use = W(idx_use, :); % L x r

    for s = 1:S
        m_joint = pack_joint_row(BetaMainTrainSm, BetaIntTrainSm, s); % 1 x D
        m0 = m_joint(idx_use) - featMu(idx_use);                    % 1 x L

        % valid dims
        ok = ~isnan(m0(:)) & all(~isnan(W_use),2) & ~isnan(sw);
        if sum(ok) < r
            warning('Subject %d: not enough valid dims to infer A (valid=%d).', s, sum(ok));
            continue;
        end

        % weighted system
        Ww  = (W_use(ok,:) .* sw(ok));         % each row scaled by sqrt(weight)
        mw0 = (m0(ok)'    .* sw(ok));          % L x 1

        a_col = Ww \ mw0;                      % r x 1
        A(s,:) = a_col(:)';
    end
end

function row = pack_joint_row(BetaMain, BetaInt, iSub)
    % Pack JOINT row in the SAME order as packJointBetas():
    % [b0(1:K), bV(1:K), bP(1:K), bR(1:K), bC(1:K), INT1(1:K), ..., INTn(1:K)]

    [~,K,~] = size(BetaMain);
    nInt = size(BetaInt,3);

    b0 = squeeze(BetaMain(iSub,:,1));
    bV = squeeze(BetaMain(iSub,:,2));
    bP = squeeze(BetaMain(iSub,:,3));
    bR = squeeze(BetaMain(iSub,:,4));
    bC = squeeze(BetaMain(iSub,:,5));

    row = [b0(:)', bV(:)', bP(:)', bR(:)', bC(:)'];

    for j = 1:nInt
        bj = squeeze(BetaInt(iSub,:,j));
        row = [row, bj(:)']; %#ok<AGROW>
    end
end

function print_corr_triplet_metrics(BetaMainTest, BetaMainAll, BetaMainGen, ...
                                    BetaIntTest,  BetaIntAll,  BetaIntGen, ...
                                    mainNames, intNames)
% Print 3 correlations (相关) per term:
%   1) corr(TEST, ALL)
%   2) corr(GEN,  ALL)
%   3) corr(GEN,  TEST)
%
% Flatten across subjects x time bins (S*K), ignoring NaNs.

    fprintf('\n--- Corr triplet metrics (相关三元组): TEST-ALL | GEN-ALL | GEN-TEST ---\n');

    % ---------- MAIN ----------
    for c = 1:5
        xT = reshape(BetaMainTest(:,:,c), [], 1);
        xA = reshape(BetaMainAll(:,:,c),  [], 1);
        xG = reshape(BetaMainGen(:,:,c),  [], 1);

        rTA = safe_corr(xT, xA);
        rGA = safe_corr(xG, xA);
        rGT = safe_corr(xG, xT);

        nTA = count_ok(xT, xA);
        nGA = count_ok(xG, xA);
        nGT = count_ok(xG, xT);

        fprintf('MAIN %-9s | T-A=%+.3f (n=%d) | G-A=%+.3f (n=%d) | G-T=%+.3f (n=%d)\n', ...
                mainNames{c}, rTA, nTA, rGA, nGA, rGT, nGT);
    end

    % ---------- INT ----------
    nInt = numel(intNames);
    for j = 1:nInt
        xT = reshape(BetaIntTest(:,:,j), [], 1);
        xA = reshape(BetaIntAll(:,:,j),  [], 1);
        xG = reshape(BetaIntGen(:,:,j),  [], 1);

        rTA = safe_corr(xT, xA);
        rGA = safe_corr(xG, xA);
        rGT = safe_corr(xG, xT);

        nTA = count_ok(xT, xA);
        nGA = count_ok(xG, xA);
        nGT = count_ok(xG, xT);

        fprintf('INT  %-9s | T-A=%+.3f (n=%d) | G-A=%+.3f (n=%d) | G-T=%+.3f (n=%d)\n', ...
                intNames{j}, rTA, nTA, rGA, nGA, rGT, nGT);
    end
end

function r = safe_corr(a, b)
% Safe correlation ignoring NaNs; returns NaN if too few points.
    ok = ~isnan(a) & ~isnan(b);
    if sum(ok) < 3
        r = NaN;
    else
        r = corr(a(ok), b(ok));
    end
end

function n = count_ok(a, b)
    n = sum(~isnan(a) & ~isnan(b));
end
