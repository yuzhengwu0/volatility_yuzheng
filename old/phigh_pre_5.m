%% resVol_RPF_fullLogit5betas.m
% Goals:
% 1) Use RPF (per subject) to generate predicted accuracy p_perf(j) for each trial
% 2) Use motion_energy to compute time-resolved residual volatility resVol_time (trial x time)
% 3) Run a logistic regression at each time bin:
%       Conf ~ f(p_j) + Correct + V + V*Correct
%    Estimate:
%       b0(t), b_P(t), b_C(t), b_V(t), b_{VC}(t)
%    Convert them to:
%       - bias kernel: Δp_high_bias(t)
%       - sensitivity kernel: Δsens(t)

clear; clc;

%% 0. Add toolboxes ------------------------------------------------------
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/Palamedes1'));

RPF_check_toolboxes;   % check RPF dependencies

%% 1. Load data & basic fields ------------------------------------------
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

% Basic fields (all original trials)
coh_all       = allStruct.rdm1_coh(:);        % coherence
resp_all      = allStruct.req_resp(:);        % 1 = right, 2 = left
correct_all   = allStruct.correct(:);         % 1 = correct, 0 = incorrect
confCont_all  = allStruct.confidence(:);      % continuous confidence (0–1)
vol_all       = allStruct.rdm1_coh_std(:);    % stimulus-level volatility
subjID_all    = allStruct.group(:);           % subject index (1/2/3)
ME_cell_all   = allStruct.motion_energy;      % N x 1 cell

% Build a single valid mask (used for both RPF and kernels)
valid = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
         ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all);

% Restrict all fields to valid trials
coh           = coh_all(valid);
resp          = resp_all(valid);
Correct       = correct_all(valid);        % used later
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = ME_cell_all(valid);        % keep cell array aligned

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% Binary confidence: high/low (>= 0.5 = high)
Conf_raw = double(confCont);
th       = 0.5;
Conf     = double(Conf_raw >= th);         % high=1, low=0

% Coherence vector (kept for possible later use)
coh_vec  = coh;

%% 2. Map volatility to condition index (for RPF) -----------------------
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end

cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;   % low volatility
cond(vol == max(vol_levels)) = 2;   % high volatility

%% 3. Run RPF to generate p_perf(j) for each trial -----------------------
% p_perf(j) = predicted P(correct) from the RPF model

subj_list   = unique(subjID);
nSubj       = numel(subj_list);
p_perf_all  = nan(nTrials, 1);   % predicted accuracy for all valid trials

for iSub = 1:nSubj

    thisSub = subj_list(iSub);
    fprintf('\n=============================\n');
    fprintf('Running RPF for subject %d\n', thisSub);
    fprintf('=============================\n');

    % Trial index for this subject (within the valid-trial space)
    idxS = (subjID == thisSub);

    coh_s     = coh(idxS);
    resp_s    = resp(idxS);           % 1/2
    correct_s = Correct(idxS);        % 0/1
    conf_s    = confCont(idxS);       % continuous confidence
    cond_s    = cond(idxS);

    if isempty(coh_s)
        warning('Subject %d has no trials. Skipping.', thisSub);
        continue;
    end

    nTr = numel(coh_s);

    % ---- Build RPF trialData ----

    % resp 1/2 -> 0/1
    resp01 = resp_s - 1;   % 1 -> 0, 2 -> 1

    % True stimulus:
    % if correct: stim = resp; if wrong: stim = 1 - resp
    stim01 = resp01;
    wrong_idx = (correct_s == 0);
    stim01(wrong_idx) = 1 - resp01(wrong_idx);

    % Map continuous (0–1) confidence to 4 rating levels (1..4)
    conf_clip = conf_s;
    conf_clip(conf_clip < 0) = 0;
    conf_clip(conf_clip > 1) = 1;
    edges     = [0, 0.25, 0.5, 0.75, 1];
    rating_s  = discretize(conf_clip, edges, 'IncludedEdge', 'right');
    rating_s(isnan(rating_s)) = 4;

    % condition: 1 = low vol, 2 = high vol
    condition_s = cond_s;

    % RPF trialData struct
    trialData = struct();
    trialData.stimID    = stim01(:)';       % 1×nTr, 0/1
    trialData.response  = resp01(:)';       % 1×nTr, 0/1
    trialData.rating    = rating_s(:)';     % 1×nTr, 1..4
    trialData.correct   = correct_s(:)';    % 1×nTr, 0/1
    trialData.x         = coh_s(:)';        % 1×nTr, coherence
    trialData.condition = condition_s(:)';  % 1×nTr, 1/2
    trialData.RT        = nan(1, nTr);      % no RT

    % ---- F1: Performance psychometric function d'(coh) ----
    F1 = struct();
    F1.info.DV                     = 'd''';
    F1.info.PF                     = @RPF_scaled_Weibull;
    F1.info.padCells               = 1;
    F1.info.set_P_max_to_d_pad_max = 1;
    F1.info.x_min                  = 0;
    F1.info.x_max                  = 1;
    F1.info.x_label                = 'coherence';
    F1.info.cond_labels            = {'low volatility', 'high volatility'};

    F1 = RPF_get_F(F1.info, trialData);   % fit performance PF

    % ---- (Optional) F2: Confidence PF (not used later, kept for structure) ----
    F2 = struct();
    F2.info.DV          = 'p(high rating)';
    F2.info.DV_respCond = 'all';
    F2.info.PF          = @PAL_Weibull;
    F2.info.x_min       = 0;
    F2.info.x_max       = 1;
    F2.info.x_label     = 'coherence';
    F2.info.cond_labels = {'low volatility', 'high volatility'};
    F2.info.constrain   = [];
    F2 = RPF_get_F(F2.info, trialData);

    % ---- Use F1 to generate p_perf_trial for each trial ----
    p_perf_trial = nan(nTr, 1);   % trialwise P(correct)

    nCond = numel(F1.data);       % usually 2 conditions: low & high vol

    for c = 1:nCond
        % Trials in this volatility condition
        mask_c = (condition_s == c);
        if ~any(mask_c)
            continue;
        end

        coh_c = coh_s(mask_c);         % coherence for these trials
        x_grid = F1.data(c).x(:);      % coherence grid
        d_grid = F1.data(c).P(:);      % predicted d'(x)

        % Try exact matching coh_c to x_grid
        [~, loc] = ismember(coh_c, x_grid);

        if any(loc == 0)
            % Interpolate for unmatched values (avoid float mismatch)
            needInterp = (loc == 0);
            d_interp   = interp1(x_grid, d_grid, coh_c(needInterp), 'linear', 'extrap');
            loc(needInterp) = numel(d_grid) + 1;
            d_grid_ext = [d_grid; d_interp(:)];
            d_pred = d_grid_ext(loc);
        else
            d_pred = d_grid(loc);
        end

        % Convert d' -> P(correct) for 2AFC
        p_corr = normcdf(d_pred ./ sqrt(2));

        p_perf_trial(mask_c) = p_corr;
    end

    % Write back into the full p_perf_all vector
    p_perf_all(idxS) = p_perf_trial;

end

fprintf('Finished RPF. Valid p_perf proportion: %.3f\n', ...
        mean(~isnan(p_perf_all)));

%% 4. Compute residual volatility from motion_energy ---------------------
% Steps: sliding-window mean/std + warp to normalized time + regress STD ~ |MEAN|

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

% Warp to 0–1 normalized time axis
nBins  = 40;
t_norm = linspace(0, 1, nBins);

MEAN_norm = nan(nTrials, nBins);
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};

    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end

    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    t_orig = linspace(0, 1, nWin_tr);

    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

% For each time bin: STD ~ |MEAN| -> residual volatility
resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 10
        continue;
    end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    y_hat = Xb * beta;
    resid = y_use - y_hat;

    tmp       = nan(size(y));
    tmp(mask_b) = resid;
    resVol_mat(:, b) = tmp;
end

% Global z-score (scaling only)
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_mat = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d time bins.\n', ...
        size(resVol_mat,1), size(resVol_mat,2));

resVol_time = resVol_mat;   % N x K

%% 5. Build performance predictor f(p_j) ---------------------------------
% p_perf_all: RPF predicted accuracy per trial (0~1)

eps = 1e-4;
p_clip = min(max(p_perf_all, eps), 1-eps);   % avoid 0 or 1

% Simple choice: z-score as f(p_j)
f_perf = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

%% 6. Full logistic model per time bin ----------------------------------
% Conf ~ f(p_j) + Correct + V + V*Correct
% Outputs:
%   b0(t), b_P(t), b_C(t), b_V(t), b_{VC}(t)
% Then convert to bias & sensitivity kernels (Δp)

[N, K] = size(resVol_time);

beta_0   = nan(K,1);
beta_P   = nan(K,1);
beta_C   = nan(K,1);
beta_V   = nan(K,1);
beta_VxC = nan(K,1);

% Added: subject dummy coefficients (Subj2 / Subj3)
beta_S2  = nan(K,1);
beta_S3  = nan(K,1);

kernel_bias_dp = nan(K,1);
kernel_sens_dp = nan(K,1);

for k = 1:K
    Vk = resVol_time(:, k);

    % Valid trials: V, Conf, Correct, f_perf, subjID are all non-NaN
    mask_k = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ...
             ~isnan(f_perf) & ~isnan(subjID);
    if sum(mask_k) < 50
        continue;
    end

    y   = Conf(mask_k);         % 0/1
    C   = Correct(mask_k);      % 0/1
    Fp  = f_perf(mask_k);       % performance predictor from RPF
    V   = Vk(mask_k);
    subj_k = subjID(mask_k);    % subject IDs for these trials

    % z-score V so β_V is per +1SD volatility
    V_z = (V - mean(V)) ./ std(V);
    VxC = V_z .* C;             % interaction term: V * Correct

    % ===== Subject baseline dummy variables =====
    % Use Subj1 as baseline:
    %   S2 = 1 for Subj2 trials; S3 = 1 for Subj3 trials
    S2 = double(subj_k == 2);
    S3 = double(subj_k == 3);

    % Design matrix (glmfit adds intercept automatically)
    % Column order: f(p_j), Correct, V_z, V_z*Correct, S2, S3
    X = [Fp, C, V_z, VxC, S2, S3];

    b = glmfit(X, y, 'binomial', 'link', 'logit');

    % Indexing
    b0  = b(1);
    bP  = b(2);
    bC  = b(3);
    bV  = b(4);
    bVC = b(5);

    % Subject baseline differences
    bS2 = b(6);
    bS3 = b(7);

    beta_0(k)   = b0;
    beta_P(k)   = bP;
    beta_C(k)   = bC;
    beta_V(k)   = bV;
    beta_VxC(k) = bVC;

    beta_S2(k)  = bS2;
    beta_S3(k)  = bS3;

    % ===== Convert coefficients to ΔP(high) and Δsens =====
    % Use a representative f(p_j): f_perf is z-scored, mean ~ 0
    Fp0 = 0;               % "typical" performance
    % Weight by empirical correct rate
    Cprob = mean(C);       % P(Correct=1)

    V_hi = +1;             % +1 SD volatility
    V_lo = -1;             % -1 SD volatility

    % helper: inverse logit
    logitinv = @(z) 1 ./ (1 + exp(-z));

    % High vol (V=+1):
    % Correct = 1
    eta_C1_Vhi = b0 + bP*Fp0 + bC*1 + bV*V_hi + bVC*(V_hi*1);
    p_C1_Vhi   = logitinv(eta_C1_Vhi);

    % Correct = 0
    eta_C0_Vhi = b0 + bP*Fp0 + bC*0 + bV*V_hi + bVC*(V_hi*0);
    p_C0_Vhi   = logitinv(eta_C0_Vhi);

    % Low vol (V=-1):
    % Correct = 1
    eta_C1_Vlo = b0 + bP*Fp0 + bC*1 + bV*V_lo + bVC*(V_lo*1);
    p_C1_Vlo   = logitinv(eta_C1_Vlo);

    % Correct = 0
    eta_C0_Vlo = b0 + bP*Fp0 + bC*0 + bV*V_lo + bVC*(V_lo*0);
    p_C0_Vlo   = logitinv(eta_C0_Vlo);

    % ---- Bias kernel: effect of volatility on overall P(high) ----
    % Weighted by empirical correct probability:
    % P(high) = P(C=1)*p(high|C=1) + P(C=0)*p(high|C=0)
    pMean_hi = Cprob * p_C1_Vhi + (1 - Cprob) * p_C0_Vhi;
    pMean_lo = Cprob * p_C1_Vlo + (1 - Cprob) * p_C0_Vlo;

    kernel_bias_dp(k) = pMean_hi - pMean_lo;    % change from -1SD -> +1SD vol

    % ---- Sensitivity kernel: effect of volatility on correct-vs-error gap ----
    gap_hi = p_C1_Vhi - p_C0_Vhi;   % gap at high vol
    gap_lo = p_C1_Vlo - p_C0_Vlo;   % gap at low vol

    kernel_sens_dp(k) = gap_hi - gap_lo;        % Δgap with higher vol

end

%% 7. Plot the 5 β(t) curves (logit space) ------------------------------
figure;

subplot(5,1,1);
plot(t_norm, beta_0, '-o', 'LineWidth', 1.2); hold on;
yline(0,'k--');
xlabel('Normalized time');
ylabel('b_0(t)');
title('Intercept b_0(t): baseline log-odds');
grid on;

subplot(5,1,2);
plot(t_norm, beta_P, '-o', 'LineWidth', 1.2); hold on;
yline(0,'k--');
xlabel('Normalized time');
ylabel('b_P(t)');
title('Performance weight b_P(t): effect of f(p_j)');
grid on;

subplot(5,1,3);
plot(t_norm, beta_C, '-o', 'LineWidth', 1.2); hold on;
yline(0,'k--');
xlabel('Normalized time');
ylabel('b_C(t)');
title('Correctness weight b_C(t): effect of being correct');
grid on;

subplot(5,1,4);
plot(t_norm, beta_V, '-o', 'LineWidth', 1.2); hold on;
yline(0,'k--');
xlabel('Normalized time');
ylabel('b_V(t)');
title('Volatility main effect b_V(t): bias kernel (logit space)');
grid on;

subplot(5,1,5);
plot(t_norm, beta_VxC, '-o', 'LineWidth', 1.2); hold on;
yline(0,'k--');
xlabel('Normalized time');
ylabel('b_{V\times C}(t)');
title('Volatility × Correct interaction b_{V×C}(t): sensitivity kernel (logit space)');
grid on;

%% 8. Plot ΔP bias kernel and Δsens kernel (probability space) ----------
figure;

subplot(2,1,1);
plot(t_norm, kernel_bias_dp, '-o', 'LineWidth', 1.2); hold on;
yline(0, 'k--');
xlabel('Normalized time within trial');
ylabel('\Delta p_{bias}(high conf)');
title('Bias kernel: effect of volatility on overall P(high)');
grid on;

subplot(2,1,2);
plot(t_norm, kernel_sens_dp, '-o', 'LineWidth', 1.2); hold on;
yline(0, 'k--');
xlabel('Normalized time within trial');
ylabel('\Delta sens (gap change)');
title('Sensitivity kernel: effect of volatility on confidence gap');
grid on;

%% 9. Per-subject 5-beta kernels ---------------------------------------
% Fit the same model separately for each subject:
%   Conf ~ f(p_j) + Correct + V + V*Correct
% Save subject-specific b0(t), b_P(t), b_C(t), b_V(t), b_{VC}(t)

uniqSubj = unique(subjID);
nSubj    = numel(uniqSubj);
[~, K]   = size(resVol_time);

% Store the 5 beta(t) curves per subject
beta_0_subj   = nan(nSubj, K);
beta_P_subj   = nan(nSubj, K);
beta_C_subj   = nan(nSubj, K);
beta_V_subj   = nan(nSubj, K);
beta_VxC_subj = nan(nSubj, K);

% Also store standard errors (SE) and p-values
se_V_subj    = nan(nSubj, K);
se_VxC_subj  = nan(nSubj, K);

p_V_subj     = nan(nSubj, K);
p_VxC_subj   = nan(nSubj, K);

for s = 1:nSubj
    thisSubj = uniqSubj(s);
    fprintf('Fitting per-subject logistic kernels: Subj %d\n', thisSubj);

    % Trials for this subject
    idxS = (subjID == thisSubj);

    Conf_s    = Conf(idxS);          % Ns x 1
    Correct_s = Correct(idxS);       % Ns x 1
    f_perf_s  = f_perf(idxS);        % Ns x 1
    Vmat_s    = resVol_time(idxS,:); % Ns x K

    for k = 1:K
        Vk = Vmat_s(:, k);    % volatility at this time bin (this subject)

        mask_sk = ~isnan(Vk) & ~isnan(Conf_s) & ~isnan(Correct_s) & ~isnan(f_perf_s);
        if sum(mask_sk) < 30
            continue;         % skip if too few trials
        end

        y  = Conf_s(mask_sk);        % 0/1
        C  = Correct_s(mask_sk);     % 0/1
        Fp = f_perf_s(mask_sk);      % z-scored f(p_j)
        V  = Vk(mask_sk);

        % z-score V within (subject x time bin)
        V_z = (V - mean(V)) ./ std(V);
        VxC = V_z .* C;

        X = [Fp, C, V_z, VxC];       % glmfit adds intercept

        [b, dev, stats] = glmfit(X, y, 'binomial', 'link', 'logit');

        beta_0_subj(s,k)   = b(1);
        beta_P_subj(s,k)   = b(2);
        beta_C_subj(s,k)   = b(3);
        beta_V_subj(s,k)   = b(4);
        beta_VxC_subj(s,k) = b(5);

        % b order: 1=intercept, 2=Fp, 3=C, 4=V_z, 5=VxC
        se_V_subj(s,k)   = stats.se(4);
        se_VxC_subj(s,k) = stats.se(5);

        p_V_subj(s,k)    = stats.p(4);
        p_VxC_subj(s,k)  = stats.p(5);

    end
end

%% 10. Use the same y-axis range for all 5 beta plots -------------------
all_beta_vals = [beta_0_subj(:); beta_P_subj(:); beta_C_subj(:); ...
                 beta_V_subj(:); beta_VxC_subj(:)];

y_all_min = min(all_beta_vals, [], 'omitnan');
y_all_max = max(all_beta_vals, [], 'omitnan');

% Add a small margin for nicer display
y_range   = y_all_max - y_all_min;
y_all_min = y_all_min - 0.05 * y_range;
y_all_max = y_all_max + 0.05 * y_range;

%% 11. Plot 5-row figure for each subject (shared y-axis) ----------------
for s = 1:nSubj
    thisSubj = uniqSubj(s);

    figure;

    % 1) b0(t)
    subplot(5,1,1);
    plot(t_norm, beta_0_subj(s,:), '-o', 'LineWidth', 1.2); hold on;
    yline(0,'k--');
    ylim([y_all_min, y_all_max]);
    xlabel('Normalized time');
    ylabel('b_0(t)');
    title(sprintf('Subj %d – Intercept b_0(t)', thisSubj));
    grid on;

    % 2) b_P(t)
    subplot(5,1,2);
    plot(t_norm, beta_P_subj(s,:), '-o', 'LineWidth', 1.2); hold on;
    yline(0,'k--');
    ylim([y_all_min, y_all_max]);
    xlabel('Normalized time');
    ylabel('b_P(t)');
    title('Performance weight b_P(t): effect of f(p_j)');
    grid on;

    % 3) b_C(t)
    subplot(5,1,3);
    plot(t_norm, beta_C_subj(s,:), '-o', 'LineWidth', 1.2); hold on;
    yline(0,'k--');
    ylim([y_all_min, y_all_max]);
    xlabel('Normalized time');
    ylabel('b_C(t)');
    title('Correctness weight b_C(t): effect of being correct');
    grid on;

    % 4) b_V(t)
    subplot(5,1,4);
    plot(t_norm, beta_V_subj(s,:), '-o', 'LineWidth', 1.2); hold on;
    yline(0,'k--');
    ylim([y_all_min, y_all_max]);
    xlabel('Normalized time');
    ylabel('b_V(t)');
    title('Volatility main effect b_V(t): bias kernel (logit space)');
    grid on;

    % 5) b_{V×C}(t)
    subplot(5,1,5);
    plot(t_norm, beta_VxC_subj(s,:), '-o', 'LineWidth', 1.2); hold on;
    yline(0,'k--');
    ylim([y_all_min, y_all_max]);
    xlabel('Normalized time');
    ylabel('b_{V\times C}(t)');
    title('Volatility \times Correct interaction b_{V\times C}(t): sensitivity kernel');
    grid on;
end

% ===== Group-level significance across subjects per time bin =====
alpha = 0.05;
useFDR = false;   % set true to use FDR correction

K = numel(t_norm);

p_group_V   = nan(1,K);
p_group_VxC = nan(1,K);

for k = 1:K
    xV   = beta_V_subj(:,k);
    xVxC = beta_VxC_subj(:,k);

    xV   = xV(~isnan(xV));
    xVxC = xVxC(~isnan(xVxC));

    if numel(xV) >= 2
        [~, p] = ttest(xV, 0);
        p_group_V(k) = p;
    end
    if numel(xVxC) >= 2
        [~, p] = ttest(xVxC, 0);
        p_group_VxC(k) = p;
    end
end

if useFDR
    q_group_V   = mafdr(p_group_V,   'BHFDR', true);
    q_group_VxC = mafdr(p_group_VxC, 'BHFDR', true);
    sigBins_V   = find(q_group_V   < alpha);
    sigBins_VxC = find(q_group_VxC < alpha);
else
    sigBins_V   = find(p_group_V   < alpha);
    sigBins_VxC = find(p_group_VxC < alpha);
end

fprintf('\n===== Bin-wise significance (group-level) =====\n');
if useFDR
    fprintf('Using FDR (q < %.3f)\n', alpha);
else
    fprintf('Using uncorrected (p < %.3f)\n', alpha);
end

% ---- V main effect ----
fprintf('\n--- b_V(t): significant bins ---\n');
if isempty(sigBins_V)
    fprintf('None\n');
else
    for ii = 1:numel(sigBins_V)
        k = sigBins_V(ii);
        if useFDR
            fprintf('bin %02d | t=%.3f | p=%.6g | q=%.6g | t_norm=%.3f\n', ...
                k, t_obs(k), p_group_V(k), q_group_V(k), t_norm(k));
        else
            fprintf('bin %02d | p=%.6g | t_norm=%.3f\n', ...
                k, p_group_V(k), t_norm(k));
        end
    end
end

% ---- VxC interaction ----
fprintf('\n--- b_{VxC}(t): significant bins ---\n');
if isempty(sigBins_VxC)
    fprintf('None\n');
else
    for ii = 1:numel(sigBins_VxC)
        k = sigBins_VxC(ii);
        if useFDR
            fprintf('bin %02d | t=%.3f | p=%.6g | q=%.6g | t_norm=%.3f\n', ...
                k, t_obs(k), p_group_VxC(k), q_group_VxC(k), t_norm(k));
        else
            fprintf('bin %02d | p=%.6g | t_norm=%.3f\n', ...
                k, p_group_VxC(k), t_norm(k));
        end
    end
end

colors = lines(nSubj);

figure; hold on;

% --- Per-subject errorbar plots ---
for s = 1:nSubj
    cap = 10;
    errorbar(t_norm, beta_V_subj(s,:), se_V_subj(s,:), '-o', ...
        'Color', colors(s,:), 'LineWidth', 1.2, 'CapSize', cap);
end

yline(0,'k--');
xlabel('Normalized time');
ylabel('b_V(t)');
legend(arrayfun(@(id)sprintf('Subj %d', id), uniqSubj, ...
       'UniformOutput', false), 'Location','best');
title('Per-subject volatility main effect b_V(t) (bias kernel, logit) + SE');
grid on;

% --- Mark significant bins on the top (group-level) ---
yl = ylim;
yStar = yl(2) - 0.05*(yl(2)-yl(1));   % place markers near top
for ii = 1:numel(sigBins_V)
    k = sigBins_V(ii);
    text(t_norm(k), yStar, '*', 'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', 'FontSize', 14);
end

figure; hold on;

for s = 1:nSubj
    cap = 10;
    errorbar(t_norm, beta_VxC_subj(s,:), se_VxC_subj(s,:), '-o', ...
        'Color', colors(s,:), 'LineWidth', 1.2, 'CapSize', cap);
end

yline(0,'k--');
xlabel('Normalized time');
ylabel('b_{V\times C}(t)');
legend(arrayfun(@(id)sprintf('Subj %d', id), uniqSubj, ...
       'UniformOutput', false), 'Location','best');
title('Per-subject volatility \times correct effect b_{V\times C}(t) (sens kernel, logit) + SE');
grid on;

yl = ylim;
yStar = yl(2) - 0.05*(yl(2)-yl(1));
for ii = 1:numel(sigBins_VxC)
    k = sigBins_VxC(ii);
    text(t_norm(k), yStar, '*', 'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', 'FontSize', 14);
end

% Input: beta_VxC_subj (nSubj x K) or beta_V_subj
B = beta_VxC_subj;          % switch to beta_V_subj if needed
[nSubj, K] = size(B);

% 1) Observed t-statistics (one-sample t-test vs 0 at each time bin)
t_obs = nan(1,K);
for k = 1:K
    x = B(:,k); x = x(~isnan(x));
    if numel(x) >= 2
        [~,~,~,st] = ttest(x,0);
        t_obs(k) = st.tstat;
    end
end

% 2) Cluster-forming threshold
alpha_bin = 0.05;
t_thr = tinv(1-alpha_bin/2, nSubj-1);   % two-sided threshold

isSupra = abs(t_obs) > t_thr;

% Find contiguous clusters
cc = bwconncomp(isSupra);   % requires Image Processing Toolbox; I can provide a no-toolbox version if needed
clusters = cc.PixelIdxList;

% Observed cluster statistic: sum of |t| within each cluster (alternative: max|t|)
cluStat_obs = zeros(1, numel(clusters));
for i = 1:numel(clusters)
    idx = clusters{i};
    cluStat_obs(i) = sum(abs(t_obs(idx)));
end
maxClu_obs = max([cluStat_obs, 0]);

% 3) Permutation test: sign-flip per subject (valid under H0)
nPerm = 5000;
maxClu_perm = zeros(nPerm,1);

for p = 1:nPerm
    flips = (rand(nSubj,1) > 0.5)*2 - 1;   % +/-1
    Bp = B .* flips;  % flip entire subject time course (preserve temporal dependence)

    t_p = nan(1,K);
    for k = 1:K
        x = Bp(:,k); x = x(~isnan(x));
        if numel(x) >= 2
            [~,~,~,st] = ttest(x,0);
            t_p(k) = st.tstat;
        end
    end

    supra_p = abs(t_p) > t_thr;
    cc_p = bwconncomp(supra_p);
    maxStat = 0;
    for i = 1:numel(cc_p.PixelIdxList)
        idx = cc_p.PixelIdxList{i};
        maxStat = max(maxStat, sum(abs(t_p(idx))));
    end
    maxClu_perm(p) = maxStat;
end

% 4) Cluster-corrected p-value (max-cluster method)
p_cluster = mean(maxClu_perm >= maxClu_obs);
fprintf('Cluster-corrected p (max cluster): %.4f\n', p_cluster);

% 5) If you want p-values per cluster
p_each = ones(1, numel(clusters));
for i = 1:numel(clusters)
    p_each(i) = mean(maxClu_perm >= cluStat_obs(i));
end
disp('Cluster p-values:'); disp(p_each);

fprintf('\n===== Cluster permutation results =====\n');
if isempty(clusters)
    fprintf('No supra-threshold clusters found.\n');
else
    for i = 1:numel(clusters)
        idx = clusters{i};
        k1  = min(idx);
        k2  = max(idx);

        fprintf('Cluster %02d | bins %02d-%02d | t_norm %.3f-%.3f | cluStat=%.3f | p=%.6g', ...
            i, k1, k2, t_norm(k1), t_norm(k2), cluStat_obs(i), p_each(i));

        if p_each(i) < 0.05
            fprintf('  <-- SIGNIFICANT\n');
        else
            fprintf('\n');
        end
    end
end
