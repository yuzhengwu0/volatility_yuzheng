%% continuous_rank2_protocol_v4_JOINTint_allOrders.m
% ==========================================================
% PURPOSE:
%   (1) TRAIN main regression per time-bin:
%       zConf ~ 1 + Vz + P + R + C
%       -> BetaMainTrain(S x K x 5), store muV/sdV
%
%   (2) TRAIN interaction regression per time-bin:
%       zConf ~ 1 + Vz + P + R + C + interactions(2/3/4-way)
%       -> BetaIntTrain(S x K x nInt)
%
%   (3) Rank-2 learning OPTIONS:
%       A) MAIN-only rank2  : learn low-dim subject constants from main betas
%       B) JOINT rank2      : learn low-dim subject constants from (main + interactions) betas
%
%   (4) TEST confidence prediction:
%       - meanBeta main only
%       - fullBeta main only (oracle)
%       - rank2 MAIN-only
%       - rank2 JOINT (main + interactions)  <-- you asked for this
%
% NOTES:
%   - zConf uses logit(conf_adj) with boundary adjustment
%   - Vz normalization uses TRAIN muV/sdV per subject per time-bin
%   - With S=3, rank2 (especially JOINT) can look extremely good in TRAIN; trust TEST more
% ==========================================================

clear; clc; close all;
rng(0);



%% =========================
% USER OPTIONS
% =========================
DO_SMOOTH_BETAS = true;
SMOOTH_WIN      = 3;               % 3 or 5

INCLUDE_INTERCEPT_IN_RANK2 = true; % include intercept in main-rank2 encoding

% interaction order: 2=only pairwise(6); 3=+3-way(10); 4=+4-way(11)
MAX_INT_ORDER = 4;

% Prediction sampling over time on TEST
PRED_MODE = "window_mean";         % "random_time" or "window_mean"
WIN_RANGE = [0.20 0.80];
Nsample   = 600;

% Minimum samples per (subject,timebin) regression
minN   = 30;
sv_tol = 1e-12;

% Optional: keep your old "TRUE prediction A_main -> interactions" as an extra baseline
DO_TRUE_PREDICT_INT = false;       % set true if you still want that section

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
nSubj     = numel(subj_list);

%% =========================
% 2) Map volatility to condition index (for RPF)
% =========================
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;

%% =========================
% 3) RPF -> predicted performance p_perf_all (trial-level)
% =========================
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

%% =========================
% 4) motion_energy -> residual volatility resVol_time (trial x K)
% =========================
winLen = 10; tol = 1e-12;

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

K = 40;
t_norm = linspace(0, 1, K);

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
eps0   = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);
Pz_all = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

Cz_all = Correct - mean(Correct,'omitnan');

rt_ref  = log(rt + 1e-6);
RTz_all = (rt_ref - mean(rt_ref,'omitnan')) ./ std(rt_ref,'omitnan');

% boundary adjust then logit (per subject)
conf_adj = nan(size(confCont));
for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = subjID == s;
    y = min(max(confCont(idxS),0),1);
    N = sum(idxS);
    y2 = (y*(N-1) + 0.5) / N;   % Smithson & Verkuilen shrink
    conf_adj(idxS) = y2;
end
zConf_all = log(conf_adj ./ (1 - conf_adj));

%% =========================
% 6) Train/test split within each subject
% =========================
trainMask = false(nTrials,1);
testMask  = false(nTrials,1);

for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = find(subjID == s);
    idxS = idxS(:);

    idxS = idxS(~isnan(zConf_all(idxS)) & ~isnan(Pz_all(idxS)) & ...
                ~isnan(Cz_all(idxS))   & ~isnan(RTz_all(idxS)));

    nS = numel(idxS);
    if nS < 20
        warning('Subject %d has few valid trials (%d).', s, nS);
    end

    rp = randperm(nS);
    nTrain = max(1, round(0.70*nS));
    trainMask(idxS(rp(1:nTrain))) = true;
    testMask(idxS(rp(nTrain+1:end))) = true;
end
fprintf('Train trials: %d | Test trials: %d\n', sum(trainMask), sum(testMask));

%% =========================
% 7) MAIN regression (train only) -> BetaMainTrain + muV/sdV
% zConf ~ 1 + Vz + P + R + C
% =========================
mainNames = {'Intercept','vol','perf','rt','corr'};
nMain = numel(mainNames);

BetaMainTrain = nan(nSubj, K, nMain);
muV = nan(nSubj, K);
sdV = nan(nSubj, K);

for iSub = 1:nSubj
    s = subj_list(iSub);

    for t = 1:K
        Vk = resVol_time(:,t);

        mask = trainMask & (subjID==s) & ~isnan(Vk) & ~isnan(zConf_all) & ...
               ~isnan(Pz_all) & ~isnan(Cz_all) & ~isnan(RTz_all);

        if sum(mask) < minN, continue; end

        y = zConf_all(mask);
        P = Pz_all(mask);
        C = Cz_all(mask);
        R = RTz_all(mask);

        Vraw = Vk(mask);
        muV(iSub,t) = mean(Vraw);
        sdV(iSub,t) = std(Vraw);
        if sdV(iSub,t) < sv_tol, continue; end

        V = (Vraw - muV(iSub,t)) ./ sdV(iSub,t);

        X = [ones(sum(mask),1), V, P, R, C];
        b = X \ y;

        BetaMainTrain(iSub,t,:) = b(:);
    end
end

for iSub = 1:nSubj
    okBins = sum(~isnan(BetaMainTrain(iSub,:,1)));
    fprintf('Subject %d: fitted MAIN bins (train) = %d / %d\n', subj_list(iSub), okBins, K);
end

%% =========================
% 7b) Smooth MAIN betas across time (optional)
% =========================
BetaMainTrainSm = BetaMainTrain;
if DO_SMOOTH_BETAS
    for iSub = 1:nSubj
        for c = 1:nMain
            y = squeeze(BetaMainTrain(iSub,:,c));
            BetaMainTrainSm(iSub,:,c) = movmean(y, SMOOTH_WIN, 'omitnan');
        end
    end
end

%% =========================
% 8) TRAIN interactions regression
% zConf ~ 1 + V + P + R + C + interactions
% =========================
[intNames, nInt] = getInteractionNames(MAX_INT_ORDER);
fprintf('\nUsing interaction order up to %d -> nInt=%d\n', MAX_INT_ORDER, nInt);

BetaIntTrain = nan(nSubj, K, nInt);

for iSub = 1:nSubj
    s = subj_list(iSub);

    for t = 1:K
        Vk = resVol_time(:,t);

        mask = trainMask & (subjID==s) & ~isnan(Vk) & ~isnan(zConf_all) & ...
               ~isnan(Pz_all) & ~isnan(Cz_all) & ~isnan(RTz_all);

        if sum(mask) < minN, continue; end

        y = zConf_all(mask);
        P = Pz_all(mask);
        C = Cz_all(mask);
        R = RTz_all(mask);

        if isnan(muV(iSub,t)) || isnan(sdV(iSub,t)) || sdV(iSub,t)<sv_tol
            continue;
        end
        V = (Vk(mask) - muV(iSub,t)) ./ sdV(iSub,t);

        Xint = buildInteractionsRow(P, V, R, C, MAX_INT_ORDER); % (N x nInt)

        X = [ones(sum(mask),1), V, P, R, C, Xint];
        b = X \ y;

        % only store interaction coefficients
        BetaIntTrain(iSub,t,:) = b(6 : (5+nInt));
    end
end

BetaIntTrainSm = BetaIntTrain;
if DO_SMOOTH_BETAS
    for iSub = 1:nSubj
        for j = 1:nInt
            tmp = squeeze(BetaIntTrain(iSub,:,j));
            BetaIntTrainSm(iSub,:,j) = movmean(tmp, SMOOTH_WIN, 'omitnan');
        end
    end
end

%% =========================
% 9) Rank-2 learning (A) MAIN-only and (B) JOINT(main+int)
% =========================
S = nSubj;
r = 2;

% ---- (A) MAIN-only rank2 ----
[M_main, mainPackInfo] = packMainBetas(BetaMainTrainSm, INCLUDE_INTERCEPT_IN_RANK2);
[A_mainRank2, W_mainRank2, mu_mainFeat, Mhat_main] = rank2_svd(M_main, r);

BetaMainHat_rank2 = unpackMainBetas(Mhat_main, mainPackInfo, K, S, INCLUDE_INTERCEPT_IN_RANK2, BetaMainTrainSm);

fprintf('\n=== MAIN-only rank2 subject constants A_mainRank2 ===\n');
for iSub = 1:S
    fprintf('Subject %d: c1=%+.4f  c2=%+.4f\n', subj_list(iSub), A_mainRank2(iSub,1), A_mainRank2(iSub,2));
end

% ---- (B) JOINT rank2: main + interactions ----
[M_joint, jointInfo] = packJointBetas(BetaMainTrainSm, BetaIntTrainSm, INCLUDE_INTERCEPT_IN_RANK2);
[A_joint, W_joint, mu_jointFeat, Mhat_joint] = rank2_svd(M_joint, r);

[BetaMainHat_joint, BetaIntHat_joint] = unpackJointBetas(Mhat_joint, jointInfo, K, S, INCLUDE_INTERCEPT_IN_RANK2, BetaMainTrainSm);

fprintf('\n=== JOINT rank2 subject constants A_joint (main+int together) ===\n');
for iSub = 1:S
    fprintf('Subject %d: c1=%+.4f  c2=%+.4f\n', subj_list(iSub), A_joint(iSub,1), A_joint(iSub,2));
end

%% =========================
% 10) Optional: TRUE prediction baseline (A_main from MAIN only -> predict interactions)
% =========================
BetaIntHat_truePred = nan(S,K,nInt);
if DO_TRUE_PREDICT_INT
    fprintf('\n=== TRUE-PRED baseline: A_mainRank2 -> predict interactions ===\n');
    BetaIntHat_truePred = truePredictInteractions(A_mainRank2, BetaIntTrainSm);
end


%% =========================
% 17) TRUE GENERATIVE DECODE (LOSO):
%     Learn joint basis W from OTHER subjects only,
%     estimate A_holdout using ONLY MAIN betas of holdout subject,
%     then generate holdout subject's FULL beta curves (main+int),
%     and compare generated interaction betas vs actual.
% =========================
fprintf('\n=== LOSO GENERATIVE DECODE: learn W from others, use MAIN to infer A, generate INT betas ===\n');

r = 2; % rank

% ---- settings: what to use to infer A of holdout subject ----
USE_MAIN_ONLY_FOR_A = true;    % recommended: avoid "seeing" interactions
INFER_A_TIME_MODE   = "all";   % "all" or "window"
A_WIN_RANGE         = [0.20 0.80]; % used if INFER_A_TIME_MODE="window"

% ---- build joint vector layout (must match your joint rank2 layout) ----
% Here we assume JOINT vector = [MAIN(5*K), INT(nInt*K)]
% MAIN order same as you used in M: b0,bvol,bperf,brt,bcorr each length K
% INT order j=1..nInt each length K

D_main  = 5*K;
D_int   = nInt*K;
D_joint = D_main + D_int;

% helper: pack one subject into joint row vector (1 x D_joint)
pack_joint = @(iSub) local_pack_joint(iSub, BetaMainTrainSm, BetaIntTrainSm, K, nInt);

% --- indices for MAIN dims inside joint vector ---
idx_main_all = 1:D_main;

% optionally restrict A inference to a time window
if INFER_A_TIME_MODE == "window"
    idxWin = find(t_norm>=A_WIN_RANGE(1) & t_norm<=A_WIN_RANGE(2));
    if isempty(idxWin), error('A_WIN_RANGE produced empty idxWin'); end

    % MAIN dims are 5 blocks of length K
    idx_main_use = [ ...
        idxWin, ...
        (K + idxWin), ...
        (2*K + idxWin), ...
        (3*K + idxWin), ...
        (4*K + idxWin) ...
    ];
else
    idx_main_use = idx_main_all;
end

% container to store LOSO-generated interaction betas
BetaIntHat_LOSO = nan(S, K, nInt);
A_LOSO          = nan(S, r);

for ihold = 1:S
    % ---- 1) training subjects = all except holdout ----
    trainSubs = setdiff(1:S, ihold);

    % ---- 2) build M_train (nTrain x D_joint) from other subjects ----
    M_train = nan(numel(trainSubs), D_joint);
    for ii = 1:numel(trainSubs)
        M_train(ii,:) = pack_joint(trainSubs(ii));
    end

    % ---- 3) impute + center using TRAIN subjects ONLY ----
    M_train_imp = imputeByFeatureMean(M_train);
    featMu_tr   = mean(M_train_imp, 1);
    M_train0    = M_train_imp - featMu_tr;

    % ---- 4) learn basis from TRAIN only ----
    [Utr, Sigtr, Vtr] = svd(M_train0, 'econ'); %#ok<ASGLU>
    W_tr = Vtr(:, 1:r);      % (D_joint x r)  basis (your "phi" flattened)

    % ---- 5) infer holdout A using ONLY MAIN dims (recommended) ----
    m_hold = pack_joint(ihold);      % (1 x D_joint) actual betas of holdout (we will only USE main dims)
    m0_hold_main = m_hold(idx_main_use) - featMu_tr(idx_main_use);  % center using train mean
    W_main = W_tr(idx_main_use, :); % (Dobs x r)

    % drop NaNs so we don't solve on missing bins
    ok = ~isnan(m0_hold_main(:)) & all(~isnan(W_main),2);
    if sum(ok) < r
        warning('Holdout sub %d: not enough valid MAIN dims to infer A (valid=%d). Skipping.', ihold, sum(ok));
        continue;
    end

    % Solve: m0' ≈ W * a'  => a' = W \ m0'
    fprintf('size(W_main(ok,:)) = [%d %d]\n', size(W_main(ok,:),1), size(W_main(ok,:),2));
    fprintf('size(m0_hold_main(ok)) = [%d %d]\n', size(m0_hold_main(ok),1), size(m0_hold_main(ok),2));
    
    ok = ~isnan(m0_hold_main(:)) & all(~isnan(W_main),2);

    a_col = W_main(ok,:) \ m0_hold_main(ok)';   % (r x 1)
    a_row = a_col(:)';                          % (1 x r)                  % (1 x r)
    A_LOSO(ihold,:) = a_row;

    % ---- 6) generate FULL joint vector for holdout ----
    m0_hat = a_row * W_tr';        % (1 x D_joint) centered prediction
    m_hat  = m0_hat + featMu_tr;   % add back train mean

    % ---- 7) decode generated interaction betas for holdout ----
    % INT block starts at D_main+1
    ptr = D_main;
    for j = 1:nInt
        BetaIntHat_LOSO(ihold,:,j) = m_hat(ptr + (1:K));
        ptr = ptr + K;
    end
end

% ---- report A_LOSO ----
fprintf('\n=== LOSO inferred A (from MAIN only) ===\n');
for iSub = 1:S
    fprintf('Subject %d: a1=%+.4f  a2=%+.4f\n', subj_list(iSub), A_LOSO(iSub,1), A_LOSO(iSub,2));
end

%% ---- (A) Plot timecourses: actual vs LOSO-generated (per interaction) ----

nC = 3; nR = ceil(nInt/nC);
figure('Color','w','Position',[180 120 1400 820]);
for j = 1:nInt
    subplot(nR,nC,j); hold on; grid on; box off;

    for iSub = 1:S
        y_act = squeeze(BetaIntTrainSm(iSub,:,j));
        y_hat = squeeze(BetaIntHat_LOSO(iSub,:,j));
        plot(t_norm, y_act, '-', 'LineWidth',1.2);   % actual
        plot(t_norm, y_hat, ':', 'LineWidth',2.0);   % generated
    end

    yline(0,'k--','LineWidth',0.8);
    title(['LOSO generate ' intNames{j} ' | solid=actual, dotted=generated']);
    xlabel('Normalized time (0–1)');
    ylabel(['\beta_{' intNames{j} '}']);
end

%% ---- (B) Scatter: actual vs LOSO-generated (per interaction) ----
figure('Color','w','Position',[200 140 1400 820]);
for j = 1:nInt
    subplot(nR,nC,j); hold on; grid on; box off;

    a = reshape(BetaIntTrainSm(:,:,j), [], 1);
    h = reshape(BetaIntHat_LOSO(:,:,j), [], 1);
    ok = ~isnan(a) & ~isnan(h);

    if sum(ok) < 3
        title(sprintf('%s: n<3', intNames{j}));
        continue;
    end

    scatter(a(ok), h(ok), 10, 'filled');
    lo = min([a(ok); h(ok)]); hi = max([a(ok); h(ok)]);
    plot([lo hi],[lo hi],'k--','LineWidth',1);

    rval = corr(a(ok), h(ok), 'type','Pearson');
    signMatch = mean(sign(a(ok)) == sign(h(ok)));

    title(sprintf('%s: r=%.3f | sign=%.2f | n=%d', intNames{j}, rval, signMatch, sum(ok)));
    xlabel('Actual \beta (regression)');
    ylabel('Generated \beta (LOSO)');
end


%% =========================
% 11) Baselines on TEST
% =========================
% meanBeta main only (per subject, averaged across time)
BetaMain_mean = nan(S, nMain);
for iSub = 1:S
    for c = 1:nMain
        BetaMain_mean(iSub,c) = mean(BetaMainTrainSm(iSub,:,c), 'omitnan');
    end
end

% oracle full main only (time-resolved)
BetaMain_full = BetaMainTrainSm;

% oracle full interactions (time-resolved)
BetaInt_full  = BetaIntTrainSm;

%% =========================
% 12) Prepare TEST samples
% =========================
testIdx = find(testMask & ~isnan(zConf_all) & ~isnan(Pz_all) & ~isnan(Cz_all) & ~isnan(RTz_all));
if isempty(testIdx), error('No valid test trials found.'); end

rp = randperm(numel(testIdx));
Nsample = min(Nsample, numel(testIdx));
testIdx = testIdx(rp(1:Nsample));

if PRED_MODE == "random_time"
    tPick = randi(K, Nsample, 1);
elseif PRED_MODE == "window_mean"
    idxWin = find(t_norm>=WIN_RANGE(1) & t_norm<=WIN_RANGE(2));
    if isempty(idxWin), error('WIN_RANGE produced empty idxWin'); end
else
    error('Unknown PRED_MODE');
end

conf_true = min(max(confCont(testIdx),0),1);
sigmoid = @(z) 1./(1+exp(-z));

%% =========================
% 13) TEST prediction loops (compare models)
% =========================
pred_meanMain  = nan(Nsample,1);
pred_fullMain  = nan(Nsample,1);
pred_rank2Main = nan(Nsample,1);
pred_rank2Joint= nan(Nsample,1);
pred_fullJoint = nan(Nsample,1);  % oracle main+int (upper bound)
pred_truePred  = nan(Nsample,1);  % optional

for i = 1:Nsample
    tr = testIdx(i);
    s  = subjID(tr);
    iSub = find(subj_list==s,1,'first');
    if isempty(iSub), continue; end

    P = Pz_all(tr);
    C = Cz_all(tr);
    R = RTz_all(tr);

    if PRED_MODE == "random_time"
        tSet = tPick(i);
    else
        tSet = idxWin(:)'; % list of t
    end

    z_mean_list   = [];
    z_full_list   = [];
    z_r2main_list = [];
    z_r2joint_list= [];
    z_fulljoint_list = [];
    z_truepred_list  = [];

    for t = tSet
        Vk = resVol_time(tr,t);
        if isnan(Vk), continue; end

        if isnan(muV(iSub,t)) || isnan(sdV(iSub,t)) || sdV(iSub,t)<sv_tol
            continue;
        end
        Vz = (Vk - muV(iSub,t)) ./ sdV(iSub,t);

        % ---------- (1) meanBeta main-only ----------
        b0m = BetaMain_mean(iSub,1); bVm = BetaMain_mean(iSub,2);
        bPm = BetaMain_mean(iSub,3); bRm = BetaMain_mean(iSub,4); bCm = BetaMain_mean(iSub,5);
        if ~any(isnan([b0m bVm bPm bRm bCm]))
            z_mean_list(end+1) = b0m + bVm*Vz + bPm*P + bRm*R + bCm*C; %#ok<AGROW>
        end

        % ---------- (2) full main-only (oracle) ----------
        b0f = BetaMain_full(iSub,t,1); bVf = BetaMain_full(iSub,t,2);
        bPf = BetaMain_full(iSub,t,3); bRf = BetaMain_full(iSub,t,4); bCf = BetaMain_full(iSub,t,5);
        if ~any(isnan([b0f bVf bPf bRf bCf]))
            z_full_list(end+1) = b0f + bVf*Vz + bPf*P + bRf*R + bCf*C; %#ok<AGROW>
        end

        % ---------- (3) rank2 MAIN-only ----------
        b0r = BetaMainHat_rank2(iSub,t,1); bVr = BetaMainHat_rank2(iSub,t,2);
        bPr = BetaMainHat_rank2(iSub,t,3); bRr = BetaMainHat_rank2(iSub,t,4); bCr = BetaMainHat_rank2(iSub,t,5);
        if ~any(isnan([b0r bVr bPr bRr bCr]))
            z_r2main_list(end+1) = b0r + bVr*Vz + bPr*P + bRr*R + bCr*C; %#ok<AGROW>
        end

        % ---------- interactions features ----------
        fInt = buildInteractionsRow(P, Vz, R, C, MAX_INT_ORDER); % 1 x nInt

        % ---------- (4) rank2 JOINT (main + int together) ----------
        b0j = BetaMainHat_joint(iSub,t,1); bVj = BetaMainHat_joint(iSub,t,2);
        bPj = BetaMainHat_joint(iSub,t,3); bRj = BetaMainHat_joint(iSub,t,4); bCj = BetaMainHat_joint(iSub,t,5);
        bInt_j = squeeze(BetaIntHat_joint(iSub,t,:))'; % 1 x nInt

        if ~any(isnan([b0j bVj bPj bRj bCj])) && ~any(isnan(bInt_j))
            z_r2joint_list(end+1) = (b0j + bVj*Vz + bPj*P + bRj*R + bCj*C) + sum(bInt_j .* fInt); %#ok<AGROW>
        end

        % ---------- (5) oracle JOINT (true main + true int) ----------
        bInt_true = squeeze(BetaInt_full(iSub,t,:))';
        if ~any(isnan([b0f bVf bPf bRf bCf])) && ~any(isnan(bInt_true))
            z_fulljoint_list(end+1) = (b0f + bVf*Vz + bPf*P + bRf*R + bCf*C) + sum(bInt_true .* fInt); %#ok<AGROW>
        end

        % ---------- optional TRUE-PRED baseline ----------
        if DO_TRUE_PREDICT_INT
            bInt_tp = squeeze(BetaIntHat_truePred(iSub,t,:))';
            if ~any(isnan([b0r bVr bPr bRr bCr])) && ~any(isnan(bInt_tp))
                z_truepred_list(end+1) = (b0r + bVr*Vz + bPr*P + bRr*R + bCr*C) + sum(bInt_tp .* fInt); %#ok<AGROW>
            end
        end
    end

    if ~isempty(z_mean_list),   pred_meanMain(i)   = mean(sigmoid(z_mean_list)); end
    if ~isempty(z_full_list),   pred_fullMain(i)   = mean(sigmoid(z_full_list)); end
    if ~isempty(z_r2main_list), pred_rank2Main(i)  = mean(sigmoid(z_r2main_list)); end
    if ~isempty(z_r2joint_list),pred_rank2Joint(i) = mean(sigmoid(z_r2joint_list)); end
    if ~isempty(z_fulljoint_list), pred_fullJoint(i)= mean(sigmoid(z_fulljoint_list)); end
    if DO_TRUE_PREDICT_INT && ~isempty(z_truepred_list), pred_truePred(i)= mean(sigmoid(z_truepred_list)); end
end

%% =========================
% 14) Evaluate + plots
% =========================
E_mean  = evalModel(conf_true, pred_meanMain);
E_full  = evalModel(conf_true, pred_fullMain);
E_r2m   = evalModel(conf_true, pred_rank2Main);
E_r2j   = evalModel(conf_true, pred_rank2Joint);
E_fj    = evalModel(conf_true, pred_fullJoint);

fprintf('\n=== TEST performance (mode: %s) ===\n', PRED_MODE);
fprintf('MeanBeta main only           : n=%d, r=%.3f, MSE=%.4f, MAE=%.4f, bias=%.4f\n', E_mean.n, E_mean.r, E_mean.mse, E_mean.mae, E_mean.bias);
fprintf('FullBeta main only (oracle)  : n=%d, r=%.3f, MSE=%.4f, MAE=%.4f, bias=%.4f\n', E_full.n, E_full.r, E_full.mse, E_full.mae, E_full.bias);
fprintf('Rank2 MAIN-only              : n=%d, r=%.3f, MSE=%.4f, MAE=%.4f, bias=%.4f\n', E_r2m.n,  E_r2m.r,  E_r2m.mse,  E_r2m.mae,  E_r2m.bias);
fprintf('Rank2 JOINT (main+int)       : n=%d, r=%.3f, MSE=%.4f, MAE=%.4f, bias=%.4f\n', E_r2j.n,  E_r2j.r,  E_r2j.mse,  E_r2j.mae,  E_r2j.bias);
fprintf('Full JOINT (oracle main+int) : n=%d, r=%.3f, MSE=%.4f, MAE=%.4f, bias=%.4f\n', E_fj.n,   E_fj.r,   E_fj.mse,   E_fj.mae,   E_fj.bias);

plotScatter(conf_true, pred_meanMain,    'MeanBeta main only', E_mean.r);
plotScatter(conf_true, pred_fullMain,    'FullBeta main only (oracle)', E_full.r);
plotScatter(conf_true, pred_rank2Main,   'Rank2 MAIN-only', E_r2m.r);
plotScatter(conf_true, pred_rank2Joint,  'Rank2 JOINT (main+int)', E_r2j.r);
plotScatter(conf_true, pred_fullJoint,   'Full JOINT (oracle main+int)', E_fj.r);

if DO_TRUE_PREDICT_INT
    E_tp = evalModel(conf_true, pred_truePred);
    fprintf('TRUE-PRED (A_main->int)      : n=%d, r=%.3f, MSE=%.4f, MAE=%.4f, bias=%.4f\n', E_tp.n, E_tp.r, E_tp.mse, E_tp.mae, E_tp.bias);
    plotScatter(conf_true, pred_truePred, 'TRUE-PRED baseline', E_tp.r);
end

%% =========================
% 15) Quick recon check plots (optional)
% =========================
figure('Color','w','Position',[200 200 1150 650]);
betaLabels = {'b_0','b_{vol}','b_{perf}','b_{rt}','b_{corr}'};
for c = 1:5
    subplot(3,2,c); hold on; grid on; box off;
    for iSub = 1:S
        raw = squeeze(BetaMainTrainSm(iSub,:,c));
        hatM= squeeze(BetaMainHat_rank2(iSub,:,c));
        hatJ= squeeze(BetaMainHat_joint(iSub,:,c));
        plot(t_norm, raw, '-', 'LineWidth', 1);
        plot(t_norm, hatM, '--', 'LineWidth', 1.6);
        plot(t_norm, hatJ, ':',  'LineWidth', 2.0);
    end
    xlabel('Normalized time (0–1)');
    ylabel(betaLabels{c});
    title(['MAIN beta: raw(solid) vs rank2Main(dash) vs rank2Joint(dotted) ' betaLabels{c}]);
end

fprintf('\nDone.\n');


%% ===== Plot interaction betas: actual(solid) vs JOINT learned(dotted) =====
fprintf('\n=== PLOT: JOINT rank2 learned interaction betas (subplot version) ===\n');

nC = 3;
nR = ceil(nInt / nC);

figure('Color','w','Position',[200 120 1400 820]);

for j = 1:nInt
    subplot(nR, nC, j); hold on; grid on; box off;

    for iSub = 1:S
        y_act = squeeze(BetaIntTrainSm(iSub,:,j));      % actual from regression
        y_hat = squeeze(BetaIntHat_joint(iSub,:,j));    % learned/recon from JOINT rank2

        plot(t_norm, y_act, '-',  'LineWidth',1.2);     % solid = actual
        plot(t_norm, y_hat, ':',  'LineWidth',2.0);     % dotted = learned
    end

    yline(0,'k--','LineWidth',0.8);
    xlabel('Normalized time (0–1)');
    ylabel(['\beta_{' intNames{j} '}']);
    title(['INT: ' intNames{j} ' | solid=actual, dotted=JOINT learned']);
end

%% ===== Scatter: learned vs actual per interaction (subplot version) =====
figure('Color','w','Position',[220 140 1400 820]);

for j = 1:nInt
    subplot(nR, nC, j); hold on; grid on; box off;

    y_act_all = reshape(BetaIntTrainSm(:,:,j), [], 1);
    y_hat_all = reshape(BetaIntHat_joint(:,:,j), [], 1);
    ok = ~isnan(y_act_all) & ~isnan(y_hat_all);

    if sum(ok) < 3
        title(sprintf('%s: n<3', intNames{j}));
        continue;
    end

    scatter(y_act_all(ok), y_hat_all(ok), 10, 'filled');

    lo = min([y_act_all(ok); y_hat_all(ok)]);
    hi = max([y_act_all(ok); y_hat_all(ok)]);
    plot([lo hi],[lo hi],'k--','LineWidth',1);

    rval = corr(y_act_all(ok), y_hat_all(ok), 'type','Pearson');
    signMatch = mean(sign(y_act_all(ok)) == sign(y_hat_all(ok)));

    xlabel('Actual \beta (regression)');
    ylabel('Learned \beta (JOINT rank2)');
    title(sprintf('%s: r=%.3f | sign=%.2f | n=%d', intNames{j}, rval, signMatch, sum(ok)));
end


%% =========================
% X) TEST regressions -> BetaMainTestSm, BetaIntTestSm
% =========================
BetaMainTest = nan(nSubj, K, nMain);
BetaIntTest  = nan(nSubj, K, nInt);

for iSub = 1:nSubj
    s = subj_list(iSub);

    for t = 1:K
        Vk = resVol_time(:,t);

        mask = testMask & (subjID==s) & ~isnan(Vk) & ~isnan(zConf_all) & ...
               ~isnan(Pz_all) & ~isnan(Cz_all) & ~isnan(RTz_all);

        if sum(mask) < minN, continue; end

        y = zConf_all(mask);
        P = Pz_all(mask);
        C = Cz_all(mask);
        R = RTz_all(mask);

        % use TRAIN muV/sdV normalization (important!)
        if isnan(muV(iSub,t)) || isnan(sdV(iSub,t)) || sdV(iSub,t) < sv_tol
            continue;
        end
        V = (Vk(mask) - muV(iSub,t)) ./ sdV(iSub,t);

        % ----- main-only -----
        Xmain = [ones(sum(mask),1), V, P, R, C];
        bmain = Xmain \ y;
        BetaMainTest(iSub,t,:) = bmain(:);

        % ----- interactions -----
        Xint = buildInteractionsRow(P, V, R, C, MAX_INT_ORDER);
        Xall = [ones(sum(mask),1), V, P, R, C, Xint];
        ball = Xall \ y;
        BetaIntTest(iSub,t,:) = ball(6:(5+nInt));
    end
end

% smooth (optional, match train smoothing)
BetaMainTestSm = BetaMainTest;
BetaIntTestSm  = BetaIntTest;
if DO_SMOOTH_BETAS
    for iSub = 1:nSubj
        for c = 1:nMain
            BetaMainTestSm(iSub,:,c) = movmean(squeeze(BetaMainTest(iSub,:,c)), SMOOTH_WIN, 'omitnan');
        end
        for j = 1:nInt
            BetaIntTestSm(iSub,:,j)  = movmean(squeeze(BetaIntTest(iSub,:,j)),  SMOOTH_WIN, 'omitnan');
        end
    end
end



%% =========================
% Y) Generative decode within-subject:
% infer a_s from TRAIN betas, generate full joint betas, compare to TEST betas
% =========================

% pack TRAIN joint row for subject s
[M_trainJoint, jointInfo] = packJointBetas(BetaMainTrainSm, BetaIntTrainSm, INCLUDE_INTERCEPT_IN_RANK2);
M_trainJoint_imp = imputeByFeatureMean(M_trainJoint);
featMu = mean(M_trainJoint_imp,1);
M0 = M_trainJoint_imp - featMu;

% learn basis W from TRAIN (same as rank2_svd but keep W and featMu)
[U,Sig,V] = svd(M0,'econ');
W = V(:,1:r);              % D x r
% A_train = U(:,1:r)*Sig(1:r,1:r); % not strictly needed

% ---- infer A for each subject from THEIR OWN TRAIN betas (row-wise) ----
A_infer = nan(nSubj, r);
for iSub = 1:nSubj
    m0 = M0(iSub,:); % 1 x D
    ok = ~isnan(m0(:)) & all(~isnan(W),2);
    ok = ok(:);

    % solve W(ok,:)*a ≈ m0(ok)'
    a = W(ok,:) \ m0(ok)';   % r x 1
    A_infer(iSub,:) = a';
end

% ---- generate full joint vector per subject ----
M0_hat = A_infer * W';      % S x D
M_hat  = M0_hat + featMu;   % add mean back

% ---- unpack into generated betas ----
[BetaMainHat_gen, BetaIntHat_gen] = unpackJointBetas(M_hat, jointInfo, K, nSubj, INCLUDE_INTERCEPT_IN_RANK2, BetaMainTrainSm);

% ===== subject color map (fixed across all plots) =====
subColors = lines(nSubj);   % MATLAB 内置好看的定性色
% subColors(iSub,:) 就是第 i 个 subject 的颜色

subColors = [ ...
    0.20 0.45 0.70;   % blue
    0.85 0.33 0.10;   % red
    0.47 0.67 0.19;   % green
];
%% =========================
% Z1) Plot MAIN: TEST (solid) vs GENERATED (dashed), same color per subject
% =========================
mainLabels = {'b_0','b_{vol}','b_{perf}','b_{rt}','b_{corr}'};

figure('Color','w','Position',[160 120 1400 820]);

for c = 1:5
    subplot(3,2,c); hold on; grid on; box off;

    for iSub = 1:nSubj
        col = subColors(iSub,:);

        y_test = squeeze(BetaMainTestSm(iSub,:,c));
        y_gen  = squeeze(BetaMainHat_gen(iSub,:,c));

        % TEST = solid
        plot(t_norm, y_test, '-', ...
            'Color', col, ...
            'LineWidth', 2.0);

        % GENERATED = dashed
        plot(t_norm, y_gen, '--', ...
            'Color', col, ...
            'LineWidth', 2.0);
    end

    yline(0,'k--','LineWidth',0.8);
    title(['MAIN ' mainLabels{c} ' | solid=TEST, dashed=GENERATED']);
    xlabel('Normalized time (0–1)');
    ylabel(mainLabels{c});
end


%% =========================
% Z2) Plot INTERACTIONS: TEST (solid) vs GENERATED (dashed), same color per subject
% =========================
nC = 3;
nR = ceil(nInt / nC);

figure('Color','w','Position',[180 120 1400 820]);

for j = 1:nInt
    subplot(nR,nC,j); hold on; grid on; box off;

    for iSub = 1:nSubj
        col = subColors(iSub,:);

        y_test = squeeze(BetaIntTestSm(iSub,:,j));
        y_gen  = squeeze(BetaIntHat_gen(iSub,:,j));

        % TEST = solid
        plot(t_norm, y_test, '-', ...
            'Color', col, ...
            'LineWidth', 2.0);

        % GENERATED = dashed
        plot(t_norm, y_gen, '--', ...
            'Color', col, ...
            'LineWidth', 2.0);
    end

    yline(0,'k--','LineWidth',0.8);
    title(['INT ' intNames{j} ' | solid=TEST, dashed=GENERATED']);
    xlabel('Normalized time (0–1)');
    ylabel(['\beta_{' intNames{j} '}']);
end

% TRAIN = thin dotted
plot(t_norm, y_train, ':', ...
    'Color', col, ...
    'LineWidth', 1.0);

%% ==========================================================
% ================= Helper functions =========================
% ==========================================================

function [A, W, featMu, M_hat] = rank2_svd(M, r)
    M_imp  = imputeByFeatureMean(M);
    featMu = mean(M_imp,1);
    M0     = M_imp - featMu;
    [U,Sig,V] = svd(M0,'econ');
    A = U(:,1:r) * Sig(1:r,1:r);
    W = V(:,1:r);
    M_hat = (A * W') + featMu;
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

function out = evalModel(y, yhat)
    ok = ~isnan(yhat);
    out.n = sum(ok);
    if out.n < 3
        out.r = NaN; out.mse = NaN; out.mae = NaN; out.bias = NaN;
        return;
    end
    out.r   = corr(y(ok), yhat(ok), 'type','Pearson');
    out.mse = mean((y(ok) - yhat(ok)).^2);
    out.mae = mean(abs(y(ok) - yhat(ok)));
    out.bias= mean(yhat(ok) - y(ok));
end

function plotScatter(y, yhat, ttl, rr)
    ok = ~isnan(yhat);
    figure('Color','w','Position',[200 200 650 560]);
    scatter(y(ok), yhat(ok), 18, 'filled'); hold on; grid on; box off;
    plot([0 1],[0 1],'k--','LineWidth',1);
    xlabel('True confidence (raw 0–1)');
    ylabel('Predicted confidence (0–1)');
    title(sprintf('%s | r=%.3f | n=%d', ttl, rr, sum(ok)));
end

function [names, nInt] = getInteractionNames(maxOrder)
    names2 = {'PxV','VxC','RxV','PxC','PxR','CxR'}; % 6
    names3 = {'PxVxR','PxVxC','PxRxC','VxRxC'};     % 4
    names4 = {'PxVxRxC'};                          % 1

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

% -------- packing/unpacking helpers (to avoid indexing bugs) --------

function [M_main, info] = packMainBetas(BetaMainTrainSm, includeIntercept)
    % BetaMainTrainSm: S x K x 5
    [S,K,~] = size(BetaMainTrainSm);
    if includeIntercept
        mainIdx = 1:5;
    else
        mainIdx = 2:5; % exclude intercept
    end
    nMainUse = numel(mainIdx);
    D = K * nMainUse;
    M_main = nan(S, D);

    for s = 1:S
        vec = [];
        for c = mainIdx
            tmp = squeeze(BetaMainTrainSm(s,:,c));
            vec = [vec, tmp(:)']; %#ok<AGROW>
        end
        M_main(s,:) = vec;
    end

    info.mainIdx = mainIdx;
    info.nMainUse = nMainUse;
end

function BetaMainHat = unpackMainBetas(Mhat_main, info, K, S, includeIntercept, BetaMainTrainSm)
    BetaMainHat = nan(S, K, 5);
    if ~includeIntercept
        % keep intercept as original smoothed if you excluded it from rank2
        BetaMainHat(:,:,1) = BetaMainTrainSm(:,:,1);
    end

    ptr0 = 0;
    for cpos = 1:info.nMainUse
        c = info.mainIdx(cpos);
        cols = ptr0 + (1:K);
        BetaMainHat(:,:,c) = Mhat_main(:, cols);
        ptr0 = ptr0 + K;
    end
end

function [M_joint, info] = packJointBetas(BetaMainTrainSm, BetaIntTrainSm, includeIntercept)
    [S,K,~] = size(BetaMainTrainSm);
    nInt = size(BetaIntTrainSm,3);

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

        % main block
        for c = mainIdx
            tmp = squeeze(BetaMainTrainSm(s,:,c));
            vec = [vec, tmp(:)']; %#ok<AGROW>
        end

        % interaction block (fixed order)
        for j = 1:nInt
            tmp = squeeze(BetaIntTrainSm(s,:,j));
            vec = [vec, tmp(:)']; %#ok<AGROW>
        end

        M_joint(s,:) = vec;
    end

    info.mainIdx  = mainIdx;
    info.nMainUse = nMainUse;
    info.nInt     = nInt;
end

function [BetaMainHat, BetaIntHat] = unpackJointBetas(Mhat_joint, info, K, S, includeIntercept, BetaMainTrainSm)
    BetaMainHat = nan(S, K, 5);
    BetaIntHat  = nan(S, K, info.nInt);

    if ~includeIntercept
        BetaMainHat(:,:,1) = BetaMainTrainSm(:,:,1);
    end

    ptr = 0;
    % main
    for cpos = 1:info.nMainUse
        c = info.mainIdx(cpos);
        cols = ptr + (1:K);
        BetaMainHat(:,:,c) = Mhat_joint(:, cols);
        ptr = ptr + K;
    end
    % interactions
    for j = 1:info.nInt
        cols = ptr + (1:K);
        BetaIntHat(:,:,j) = Mhat_joint(:, cols);
        ptr = ptr + K;
    end
end

function BetaIntHat = truePredictInteractions(A_main, BetaIntTrainSm)
    % replicate your old idea: linear map A_main -> interactions vector
    [S,K,nInt] = size(BetaIntTrainSm);
    r = size(A_main,2);

    Dint = K*nInt;
    Mint = nan(S, Dint);
    for s = 1:S
        vec = [];
        for j = 1:nInt
            tmp = squeeze(BetaIntTrainSm(s,:,j));
            vec = [vec, tmp(:)']; %#ok<AGROW>
        end
        Mint(s,:) = vec;
    end

    Mint_imp = imputeByFeatureMean(Mint);
    mu_int   = mean(Mint_imp,1);
    Mint0    = Mint_imp - mu_int;

    WintT     = A_main \ Mint0;     % (r x Dint)
    Mint0_hat = A_main * WintT;     % (S x Dint)
    Mint_hat  = Mint0_hat + mu_int;

    BetaIntHat = nan(S,K,nInt);
    ptr = 0;
    for j = 1:nInt
        cols = ptr + (1:K);
        BetaIntHat(:,:,j) = Mint_hat(:, cols);
        ptr = ptr + K;
    end
end

%% ===== local helper: pack joint row =====
function row = local_pack_joint(iSub, BetaMainTrainSm, BetaIntTrainSm, K, nInt)
    % row: [MAIN(5*K), INT(nInt*K)]
    b0    = squeeze(BetaMainTrainSm(iSub,:,1));
    bvol  = squeeze(BetaMainTrainSm(iSub,:,2));
    bperf = squeeze(BetaMainTrainSm(iSub,:,3));
    brt   = squeeze(BetaMainTrainSm(iSub,:,4));
    bcorr = squeeze(BetaMainTrainSm(iSub,:,5));

    row = nan(1, 5*K + nInt*K);
    row(1:K)           = b0(:)';
    row(K+(1:K))       = bvol(:)';
    row(2*K+(1:K))     = bperf(:)';
    row(3*K+(1:K))     = brt(:)';
    row(4*K+(1:K))     = bcorr(:)';

    ptr = 5*K;
    for j = 1:nInt
        tmp = squeeze(BetaIntTrainSm(iSub,:,j));
        row(ptr + (1:K)) = tmp(:)';
        ptr = ptr + K;
    end
end
