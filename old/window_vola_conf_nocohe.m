%% logit_resVol_confBias_withCoh.m
% 目标（在你原码基础上 + coherence）：
% 1) 从 all.motion_energy 读出 frame-by-frame signed evidence（右正左负）
% 2) 按 10-frame sliding window 算每个 window 的 mean evidence 和 volatility（std）
% 3) 把 mean & std 都归一化到 0–1 时间轴（40 个时间点）
% 4) 在每个时间点上，用 STD ~ |MEAN| 回归，取 residual STD = 额外 volatility
% 5) 把时间切成 Early / Mid / Late 三段，取三段 residual volatility 作为 predictor
% 6) 把 confidence 二分（high / low），先做：
%       HighConf ~ ResVol_E + ResVol_M + ResVol_L
%    再做：
%       HighConf ~ ResVol_E + ResVol_M + ResVol_L + MeanCoh
%    看在控制 mean coherence 后，ResVol 的 early/mid/late 效果还在不在。

clear; clc;

%% 0. 读入数据
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell（差值 evidence，右正左负）
nTrials       = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 0.1 读入 confidence（continuous）并二分成 high / low
% !!! 这里改成你真正的信心字段名 !!!
% 例如 allStruct.confidence / allStruct.conf / allStruct.rating 等
conf_vec = allStruct.confidence(:);   % <--- 如果名字不是 confidence，这里要改

fprintf('Confidence range (ignoring NaNs): [%.2f, %.2f]\n', ...
        min(conf_vec(~isnan(conf_vec))), max(conf_vec(~isnan(conf_vec))));

medConf   = median(conf_vec(~isnan(conf_vec)));
high_conf = conf_vec > medConf;   % 1 = high, 0 = low

fprintf('Median confidence = %.2f\n', medConf);
fprintf('High-confidence trials: %d / %d (%.1f%%)\n', ...
        sum(high_conf==1 & ~isnan(high_conf)), ...
        sum(~isnan(high_conf)), ...
        100 * sum(high_conf==1 & ~isnan(high_conf)) / sum(~isnan(high_conf)));

%% 0.2 读入 trial-level mean coherence（当 confound）
% 如果你已经有现成的每个 trial coherence，就直接拿：
%   mean_coh = allStruct.coherence(:);   或者 allStruct.coh 等
% 如果你没有，就用 stimulus 里的别的信息去算。
% 这里先按“已经有变量”的写法，你把变量名改一下就行。

% !!! 把 mean_coh 这一行改成你真实的 coherence 向量 !!!
mean_coh = allStruct.rdm1_coh(:);   % <--- 比如叫 coherence / coh / stimCoh 等

fprintf('Coherence range (ignoring NaNs): [%.3f, %.3f]\n', ...
        min(mean_coh(~isnan(mean_coh))), max(mean_coh(~isnan(mean_coh))));

%% 0.3 如果只想看 correct trial 的 confidence bias，可以在这里设 mask
if isfield(allStruct, 'correct')
    correct_mask = (allStruct.correct(:) == 1);
else
    % 如果暂时没有 correct 字段，就先把所有 trial 都当作 valid
    correct_mask = true(nTrials, 1);
end

%% 1. 对每个 trial 做 sliding window，算 mean & std（去尾巴 0）

winLen = 10;
tol    = 1e-12;

evidence_strength   = cell(nTrials, 1);   % 每个 trial: window mean
volatility_strength = cell(nTrials, 1);   % 每个 trial: window std

for tr = 1:nTrials
    frames = motion_energy{tr};   % nFrames x 1
    trace  = frames(:)';          % 1 x nFrames（可能带 padding 0）
    
    % 去掉末尾 padding 的 0：只保留到最后一个非 0 帧
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
    
    nWin    = nFrames - winLen + 1;
    ev_mean = nan(1, nWin);
    ev_std  = nan(1, nWin);
    
    for w = 1:nWin
        segment    = trace_eff(w : w + winLen - 1);
        ev_mean(w) = mean(segment);   % 这一 window 里的平均 evidence
        ev_std(w)  = std(segment);    % 这一 window 里的 volatility
    end
    
    evidence_strength{tr}   = ev_mean;
    volatility_strength{tr} = ev_std;
end

%% 2. 把 mean & std 都插值到统一的 0–1 时间轴（40 个时间点）

nBins  = 40;                       % 归一化时间上的 bin 数
t_norm = linspace(0, 1, nBins);    % 0 = trial 开始, 1 = 结束

MEAN_norm = nan(nTrials, nBins);   % trial x time
STD_norm  = nan(nTrials, nBins);   % trial x time

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};
    
    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end
    
    % 保险：长度取两者里较短那一个
    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);
    
    t_orig = linspace(0, 1, nWin_tr);   % 这个 trial 自己的 0–1 时间
    
    MEAN_norm(tr, :) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,  :) = interp1(t_orig, sd_tr, t_norm, 'linear');
end

%% 3. 在每个时间 bin 上，用 STD ~ |MEAN| 回归，拿 residual STD（额外 volatility）

resid_STD = nan(size(STD_norm));   % trial x time

for b = 1:nBins
    y = STD_norm(:, b);           % volatility
    x = abs(MEAN_norm(:, b));     % 证据强度的绝对值
    
    mask = ~isnan(y) & ~isnan(x);
    if sum(mask) < 10
        continue;
    end
    
    Xb    = [ones(sum(mask),1), x(mask)];   % [常数项, |mean|]
    betab = Xb \ y(mask);                   % 拟合 y = b0 + b1 * |mean|
    
    y_hat = Xb * betab;                     % 给定 |mean| “正常”应该有的 std
    
    tmp       = nan(size(y));
    tmp(mask) = y(mask) - y_hat;            % residual = 真实 std - 正常 std
    resid_STD(:, b) = tmp;                  % 这就是额外 volatility
end

%% 4. 把时间分成 Early / Mid / Late 三段，取 residual volatility 的平均值

early_idx = 1:13;       % 大约 0 ~ 0.33
mid_idx   = 14:26;      % 大约 0.33 ~ 0.66
late_idx  = 27:40;      % 大约 0.66 ~ 1

ResVol_E = mean(resid_STD(:, early_idx), 2, 'omitnan');
ResVol_M = mean(resid_STD(:, mid_idx),   2, 'omitnan');
ResVol_L = mean(resid_STD(:, late_idx),  2, 'omitnan');

%% 5. 构造 logistic 回归：先不加 coherence，再加 coherence

y = high_conf;   % 0 = low confidence, 1 = high confidence

% 有效 trial：ResVol 不 NaN、confidence 不 NaN、coherence 不 NaN、并且在 correct_mask 里
valid_mask = all(~isnan([ResVol_E, ResVol_M, ResVol_L]), 2) ...
             & ~isnan(y) ...
             & ~isnan(mean_coh) ...
             & correct_mask;

fprintf('Using %d valid trials for logistic regression (confidence bias).\n', ...
        sum(valid_mask));

if ~any(valid_mask)
    error('No valid trials left after removing NaNs.');
end

% predictor 矩阵
X_resVol = [ResVol_E(valid_mask), ResVol_M(valid_mask), ResVol_L(valid_mask)];
coh_vec  = mean_coh(valid_mask);
y_valid  = y(valid_mask);

%% 5.1 模型1：只用 residual volatility（和你原来的一样）

[b1, dev1, stats1] = glmfit(X_resVol, y_valid, 'binomial', 'link', 'logit');

pred_names1 = {'Intercept','ResVol_Early','ResVol_Mid','ResVol_Late'};

fprintf('\n=== Logistic regression WITHOUT coherence ===\n');
for k = 1:numel(pred_names1)
    fprintf('%15s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        pred_names1{k}, b1(k), stats1.se(k), stats1.p(k));
end

%% 5.2 模型2：把 mean coherence 当 confound 加进去

X_withCoh = [X_resVol, coh_vec];  % [ResVol_E, ResVol_M, ResVol_L, MeanCoh]

[b2, dev2, stats2] = glmfit(X_withCoh, y_valid, 'binomial', 'link', 'logit');

pred_names2 = {'Intercept','ResVol_Early','ResVol_Mid','ResVol_Late','MeanCoh'};

fprintf('\n=== Logistic regression WITH coherence as covariate ===\n');
for k = 1:numel(pred_names2)
    fprintf('%15s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        pred_names2{k}, b2(k), stats2.se(k), stats2.p(k));
end

%% 5.3 比较两个模型：加入 coherence 有没有显著提高拟合（LRT）

% deviance 差异 + 1 个 df（因为只多了一个 MeanCoh）
LR_stat = dev1 - dev2;         % 大模型 dev 更小
p_LR    = 1 - chi2cdf(LR_stat, 1);

fprintf('\n=== Likelihood-ratio test: adding coherence ===\n');
fprintf('Deviance(noCoh) = %.3f,  Deviance(withCoh) = %.3f,  ΔDev = %.3f,  p = %.4g\n', ...
        dev1, dev2, LR_stat, p_LR);

%% 5.4 看 residual volatility 和 coherence 本身相关多强（诊断一下）

[rE, pE] = corr(ResVol_E(valid_mask), coh_vec);
[rM, pM] = corr(ResVol_M(valid_mask), coh_vec);
[rL, pL] = corr(ResVol_L(valid_mask), coh_vec);

fprintf('\n=== Corr(ResVol, MeanCoh) on valid trials ===\n');
fprintf('Early: r = %.3f, p = %.3g\n', rE, pE);
fprintf(' Mid : r = %.3f, p = %.3g\n', rM, pM);
fprintf(' Late: r = %.3f, p = %.3g\n', rL, pL);


%% 5.5 再把 ResVol_E/M/L 里「coherence 的影响」回归掉，得到纯粹的 vola 残差

mask   = valid_mask;          % 只在 valid trial 上做
coh_sub = coh_vec;            % coh_vec = mean_coh(valid_mask)，已经是 valid 的

% Early: ResVol_E ~ MeanCoh
yE = ResVol_E(mask);
XE = [ones(sum(mask),1), coh_sub];    % [常数, MeanCoh]
bE = XE \ yE;                         % 最小二乘回归
yE_hat = XE * bE;
ResVol_E_residCoh        = nan(size(ResVol_E));
ResVol_E_residCoh(mask)  = yE - yE_hat;

% Mid: ResVol_M ~ MeanCoh
yM = ResVol_M(mask);
XM = [ones(sum(mask),1), coh_sub];
bM = XM \ yM;
yM_hat = XM * bM;
ResVol_M_residCoh        = nan(size(ResVol_M));
ResVol_M_residCoh(mask)  = yM - yM_hat;

% Late: ResVol_L ~ MeanCoh
yL = ResVol_L(mask);
XL = [ones(sum(mask),1), coh_sub];
bL = XL \ yL;
yL_hat = XL * bL;
ResVol_L_residCoh        = nan(size(ResVol_L));
ResVol_L_residCoh(mask)  = yL - yL_hat;

fprintf('\n=== Regression of ResVol on MeanCoh (betas) ===\n');
fprintf('Early:  b0 = %.3e, b1 = %.3e\n', bE(1), bE(2));
fprintf('Mid  :  b0 = %.3e, b1 = %.3e\n', bM(1), bM(2));
fprintf('Late :  b0 = %.3e, b1 = %.3e\n', bL(1), bL(2));

%% 5.6 检查：在减掉 coherence 之后，ResVol_*_residCoh 和 MeanCoh 还相关吗？

[rE2, pE2] = corr(ResVol_E_residCoh(mask), coh_sub);
[rM2, pM2] = corr(ResVol_M_residCoh(mask), coh_sub);
[rL2, pL2] = corr(ResVol_L_residCoh(mask), coh_sub);

fprintf('\n=== Corr(ResVol_residCoh, MeanCoh) on valid trials ===\n');
fprintf('Early (after residualizing): r = %.3f, p = %.3g\n', rE2, pE2);
fprintf(' Mid  (after residualizing): r = %.3f, p = %.3g\n', rM2, pM2);
fprintf(' Late (after residualizing): r = %.3f, p = %.3g\n', rL2, pL2);

%% 6. 画图：已经「去除了 coherence 线性影响」的 residual volatility vs confidence

idxHigh = (y == 1) & valid_mask;
idxLow  = (y == 0) & valid_mask;

figure;
subplot(3,1,1);
histogram(ResVol_E_residCoh(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_E_residCoh(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('ResVol Early (residualized for coherence)');
ylabel('Density');
title('Early residual volatility (coh-partialed) vs confidence');
grid on;

subplot(3,1,2);
histogram(ResVol_M_residCoh(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_M_residCoh(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('ResVol Mid (residualized for coherence)');
ylabel('Density');
title('Mid residual volatility (coh-partialed) vs confidence');
grid on;

subplot(3,1,3);
histogram(ResVol_L_residCoh(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_L_residCoh(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('ResVol Late (residualized for coherence)');
ylabel('Density');
title('Late residual volatility (coh-partialed) vs confidence');
grid on;

%% 6. 画图：高/低信心 trial 的 residual volatility 分布（跟你原来的一样）

idxHigh = (y == 1) & valid_mask;
idxLow  = (y == 0) & valid_mask;

figure;
subplot(3,1,1);
histogram(ResVol_E(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_E(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('Residual volatility (Early)');
ylabel('Density');
title('Early residual volatility vs confidence');
grid on;

subplot(3,1,2);
histogram(ResVol_M(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_M(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('Residual volatility (Mid)');
ylabel('Density');
title('Mid residual volatility vs confidence');
grid on;

subplot(3,1,3);
histogram(ResVol_L(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_L(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('Residual volatility (Late)');
ylabel('Density');
title('Late residual volatility vs confidence');
grid on;

