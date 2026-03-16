%% resVol_RPF_fullKernel.m
% 目标：
% 1) 用 RPF（per subject）为每个 trial 生成预测正确率 p_perf(j)
% 2) 用 motion_energy 算 time-resolved residual volatility resVol_time(trial x time)
% 3) 对每个 time bin 做 logistic 回归：
%       Conf ~ f(p_j) + Correct + V + V*Correct
%    得到：
%       - bias kernel: Δp_high_bias(t)
%       - sensitivity kernel: Δsens(t)

clear; clc;

%% 0. Add toolboxes ------------------------------------------------------
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/Palamedes1'));

RPF_check_toolboxes;   % 检查 RPF 依赖

%% 1. Load data & basic fields ------------------------------------------
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

% 基本字段（原始全 trial）
coh_all    = allStruct.rdm1_coh(:);        % coherence
resp_all   = allStruct.req_resp(:);        % 1 = right, 2 = left
correct_all= allStruct.correct(:);         % 1 = correct, 0 = incorrect
confCont_all = allStruct.confidence(:);    % 0–1 连续 confidence
vol_all    = allStruct.rdm1_coh_std(:);    % volatility (刺激层面的)
subjID_all = allStruct.group(:);           % subject index (1/2/3)
ME_cell_all= allStruct.motion_energy;      % N x 1 cell

% 有些字段可能有 NaN，先做一个 valid mask（用于 RPF + kernel 一致）
valid = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
         ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all);

% 把所有字段都 restrict 到 valid trial
coh        = coh_all(valid);
resp       = resp_all(valid);
Correct    = correct_all(valid);       % 用于后面 kernel
confCont   = confCont_all(valid);
vol        = vol_all(valid);
subjID     = subjID_all(valid);
motion_energy = ME_cell_all(valid);    % cell 数组也要对齐

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% binary confidence：高/低（>=0.5 = 高）
Conf_raw = double(confCont);
th       = 0.5;
Conf     = double(Conf_raw >= th);     % 高=1, 低=0

% coherence 向量（为了后面如果你要用）
coh_vec  = coh;

%% 2. Map volatility to condition index (for RPF) -----------------------
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end

cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;   % low volatility
cond(vol == max(vol_levels)) = 2;   % high volatility

%% 3. 用 RPF：为每个 trial 生成 p_perf(j) -------------------------------
% p_perf(j) = 这个 trial 在 RPF 模型下的预测正确率（performance）

subj_list   = unique(subjID);
nSubj       = numel(subj_list);
p_perf_all  = nan(nTrials, 1);   % 最终所有 valid trial 的 p_j

for iSub = 1:nSubj

    thisSub = subj_list(iSub);
    fprintf('\n=============================\n');
    fprintf('Running RPF for subject %d\n', thisSub);
    fprintf('=============================\n');

    % 这个被试的 trial index（在 valid 之后的空间里）
    idxS = (subjID == thisSub);

    coh_s     = coh(idxS);
    resp_s    = resp(idxS);           % 1/2
    correct_s = Correct(idxS);        % 0/1
    conf_s    = confCont(idxS);       % 连续 confidence
    cond_s    = cond(idxS);

    if isempty(coh_s)
        warning('Subject %d has no trials. Skipping.', thisSub);
        continue;
    end

    nTr = numel(coh_s);

    % ---- 构造 RPF 所需的 trialData ----

    % resp 1/2 → 0/1
    resp01 = resp_s - 1;   % 1 -> 0, 2 -> 1

    % true stimulus: correct → stim = resp; wrong → stim = 1 - resp
    stim01 = resp01;
    wrong_idx = (correct_s == 0);
    stim01(wrong_idx) = 1 - resp01(wrong_idx);

    % 连续 0–1 confidence → 4 档 rating (1..4)
    conf_clip = conf_s;
    conf_clip(conf_clip < 0) = 0;
    conf_clip(conf_clip > 1) = 1;
    edges     = [0, 0.25, 0.5, 0.75, 1];
    rating_s  = discretize(conf_clip, edges, 'IncludedEdge', 'right');
    rating_s(isnan(rating_s)) = 4;

    % condition: 1 = low vol, 2 = high vol
    condition_s = cond_s;

    % RPF trialData
    trialData = struct();
    trialData.stimID    = stim01(:)';       % 1×nTr, 0/1
    trialData.response  = resp01(:)';       % 1×nTr, 0/1
    trialData.rating    = rating_s(:)';     % 1×nTr, 1..4
    trialData.correct   = correct_s(:)';    % 1×nTr, 0/1
    trialData.x         = coh_s(:)';        % 1×nTr, coherence
    trialData.condition = condition_s(:)';  % 1×nTr, 1/2
    trialData.RT        = nan(1, nTr);      % 没有 RT

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

    F1 = RPF_get_F(F1.info, trialData);   % 拟合 performance PF

    % ---- F2: Confidence psychometric function（暂时只用 F1 来算 p_j）----
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

    % （如需 RPF 的 R，可继续算；这里主要是用 F1 来拿 performance）
    % P1_LB = [];
    % P1_UB = [];
    % R     = RPF_get_R(F1, F2, P1_LB, P1_UB);

    % ---- 利用 F1 为这个被试的每个 trial 生成 p_perf_trial ----
    p_perf_trial = nan(nTr, 1);   % 每个 trial 的 p(correct)

    nCond = numel(F1.data);       % 一般是 2 个：low & high vol

    for c = 1:nCond
        % 当前 vol 条件下的 trial
        mask_c = (condition_s == c);
        if ~any(mask_c)
            continue;
        end

        coh_c = coh_s(mask_c);         % 这些 trial 的 coherence
        x_grid = F1.data(c).x(:);      % 该条件下的 coh 网格
        d_grid = F1.data(c).P(:);      % 对应的 d'(x)

        % 方法 A：如果 coh_c 刚好只取 x_grid 里的离散水平 → 直接匹配
        [~, loc] = ismember(coh_c, x_grid);

        % 如果有没匹配上的，用插值（防止浮点问题）
        if any(loc == 0)
            % 对没匹配上的那部分用插值
            needInterp = (loc == 0);
            d_interp   = interp1(x_grid, d_grid, coh_c(needInterp), 'linear', 'extrap');
            loc(needInterp) = numel(d_grid) + 1;
            d_grid_ext = [d_grid; d_interp(:)];
            d_pred = d_grid_ext(loc);
        else
            d_pred = d_grid(loc);
        end

        % d' → p(correct) （2AFC）
        p_corr = normcdf(d_pred ./ sqrt(2));

        p_perf_trial(mask_c) = p_corr;
    end

    % 把这个被试的 p_perf_trial 塞回全体向量 p_perf_all
    p_perf_all(idxS) = p_perf_trial;

end

fprintf('Finished RPF. Valid p_perf proportion: %.3f\n', ...
        mean(~isnan(p_perf_all)));

%% 4. 用 motion_energy 算 residual volatility ---------------------------
% 这一段基本是你原来的 sliding window + warp + 回归 STD~|MEAN|

% motion_energy 已经被 valid mask 限制到 nTrials 大小
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

% warp 到 0–1 的 normalized time 轴
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

% 每个 time bin: STD ~ |MEAN| → residual volatility
resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask = ~isnan(y) & ~isnan(x1);
    if sum(mask) < 10
        continue;
    end

    X     = [ones(sum(mask),1), x1(mask)];
    y_use = y(mask);

    beta  = X \ y_use;
    y_hat = X * beta;
    resid = y_use - y_hat;

    tmp       = nan(size(y));
    tmp(mask) = resid;
    resVol_mat(:, b) = tmp;
end

% 全局 z-score（只是 scale，不改形状）
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_mat = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d time bins.\n', ...
        size(resVol_mat,1), size(resVol_mat,2));

resVol_time = resVol_mat;   % N x K

%% 5. 准备 performance predictor f(p_j) ---------------------------------
% p_perf_all: 每个 trial 的 RPF 预测正确率（0~1）
eps = 1e-4;
p_clip = min(max(p_perf_all, eps), 1-eps);   % 避免 0 或 1

% 最简单：z-score 作为 f(p_j)
f_perf = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

%% 6. FULL logistic model per time bin ----------------------------------
% Conf ~ f(p_j) + Correct + V + V*Correct
% 得到：
%   beta_perf(k), beta_C(k), beta_V(k), beta_VxC(k)
% 并把这些系数转成 bias kernel & sensitivity kernel（Δp）

[N, K] = size(resVol_time);

beta_perf   = nan(K,1);
beta_C      = nan(K,1);
beta_V      = nan(K,1);
beta_VxC    = nan(K,1);
beta_0      = nan(K,1);

kernel_bias_dp = nan(K,1);   % Δp_high_bias(t): vol ↑1SD 时整体 p(high) 变化
kernel_sens_dp = nan(K,1);   % Δsens(t): vol ↑1SD 时 correct vs error gap 的变化

for k = 1:K
    Vk = resVol_time(:, k);

    % 有效 trial：V、Conf、Correct、f_perf 都不是 NaN
    mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(f_perf);
    if sum(mask) < 50
        continue;
    end

    y   = Conf(mask);         % 0/1
    C   = Correct(mask);      % 0/1
    Fp  = f_perf(mask);       % RPF-derived performance predictor
    V   = Vk(mask);

    % z-score V，使 β_V 以 “每 +1SD vol” 为单位
    V_z = (V - mean(V)) ./ std(V);

    VxC = V_z .* C;           % 交互项：V * Correct

    % 设计矩阵（glmfit 自动加 intercept）
    % 列顺序：f(p_j), Correct, V_z, V_z*Correct
    X = [Fp, C, V_z, VxC];

    b = glmfit(X, y, 'binomial', 'link', 'logit');

    b0   = b(1);
    bP   = b(2);
    bC   = b(3);
    bV   = b(4);
    bVC  = b(5);

    beta_0(k)   = b0;
    beta_perf(k)= bP;
    beta_C(k)   = bC;
    beta_V(k)   = bV;
    beta_VxC(k) = bVC;

    % ===== 把系数转成 “Δp(high)” 和 “Δsens” =====
    % 选一个代表性的 f(p_j)：f_perf 已 z-score，均值 ~ 0
    Fp0 = 0;               % "typical" performance
    % 用真实数据的 correct 比例做加权平均
    Cprob = mean(C);       % P(Correct=1)

    V_hi = +1;             % +1 SD volatility
    V_lo = -1;             % -1 SD volatility

    % 辅助：logit^-1
    logitinv = @(z) 1 ./ (1 + exp(-z));

    % 高 vol（V=+1）下：
    % Correct = 1
    eta_C1_Vhi = b0 + bP*Fp0 + bC*1 + bV*V_hi + bVC*(V_hi*1);
    p_C1_Vhi   = logitinv(eta_C1_Vhi);

    % Correct = 0
    eta_C0_Vhi = b0 + bP*Fp0 + bC*0 + bV*V_hi + bVC*(V_hi*0);
    p_C0_Vhi   = logitinv(eta_C0_Vhi);

    % 低 vol（V=-1）下：
    % Correct = 1
    eta_C1_Vlo = b0 + bP*Fp0 + bC*1 + bV*V_lo + bVC*(V_lo*1);
    p_C1_Vlo   = logitinv(eta_C1_Vlo);

    % Correct = 0
    eta_C0_Vlo = b0 + bP*Fp0 + bC*0 + bV*V_lo + bVC*(V_lo*0);
    p_C0_Vlo   = logitinv(eta_C0_Vlo);

    % ---- Bias kernel: 看整体 p(high) 怎么被 vol 推动 ----
    % 用真实 correct 概率加权：P(high) = P(C=1)*p(high|C=1) + P(C=0)*p(high|C=0)
    pMean_hi = Cprob * p_C1_Vhi + (1 - Cprob) * p_C0_Vhi;
    pMean_lo = Cprob * p_C1_Vlo + (1 - Cprob) * p_C0_Vlo;

    kernel_bias_dp(k) = pMean_hi - pMean_lo;    % vol 从 -1SD → +1SD 时整体 p(high) 的变化

    % ---- Sensitivity kernel: 看 correct vs error gap 怎么被 vol 改变 ----
    gap_hi = p_C1_Vhi - p_C0_Vhi;   % 高 vol 时 correct vs error 的 gap
    gap_lo = p_C1_Vlo - p_C0_Vlo;   % 低 vol 时 correct vs error 的 gap

    kernel_sens_dp(k) = gap_hi - gap_lo;   % vol ↑1SD 时 gap 的变化（Δsens）

end

%% 7. 画 bias & sensitivity kernel -------------------------------------
figure;

subplot(3,1,1);
plot(t_norm, kernel_bias_dp, '-o', 'LineWidth', 1.2); hold on;
yline(0, 'k--');
xlabel('Normalized time within trial');
ylabel('\Delta p_{bias}(high conf)');
title('Bias kernel: effect of volatility on overall P(high)');
grid on;

subplot(3,1,2);
plot(t_norm, kernel_sens_dp, '-o', 'LineWidth', 1.2); hold on;
yline(0, 'k--');
xlabel('Normalized time within trial');
ylabel('\Delta sens (gap change)');
title('Sensitivity kernel: effect of volatility on confidence gap');
grid on;

subplot(3,1,3);
plot(t_norm, beta_V, '-o'); hold on;
plot(t_norm, beta_VxC, '-o');
yline(0,'k--');
xlabel('Normalized time within trial');
ylabel('\beta_V, \beta_{V\times C}');
legend({'\beta_V (main effect)', '\beta_{V\times C} (interaction)'}, 'Location','best');
title('Logistic coefficients of volatility');
grid on;
