%% conf_kernels_and_predict.m
% 目标：
% 1) 从 motion_energy 重新算 residual volatility kernel: resVol_mat (trial x time bin)
% 2) 对每个 subject 做 time-resolved logistic regression，得到：
%      biasKernel(s,t) = β_vol(t)
%      sensKernel(s,t) = δ_vol(t)
%      confBias(s)     = overall 截距
%      gamma(s)        = Correct 的整体效应
% 3) 用这些 kernel + residual volatility 去预测每个 trial 的 p(high confidence)
% 4) 画一个 build-up 图，看随着 time bin 累加，预测能力怎么变化

clear; clc;

%% 0. 加载原始数据 -----------------------------------------------------
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';  % TODO: 改成你的路径
tmp       = load(data_path, 'all');
allStruct = tmp.all;

% 0.1 把 0~1 的 conf 变成 0/1 高/低信心
Conf_raw  = double(allStruct.confidence(:));   % 0–1 连续
th        = 0.5;                               % 阈值：>=0.5 高信心
Conf      = double(Conf_raw >= th);           % 高=1, 低=0

Correct   = double(allStruct.correct(:));      % 0/1
coh       = double(allStruct.rdm1_coh(:));     % coherence（算 residual volatility 用）
volCond   = double(allStruct.rdm1_coh_std(:)); % volatility 条件（这里不直接用在 regression 里）
subjID    = double(allStruct.group(:));        % subject ID
motion_en = allStruct.motion_energy;           % cell, 每个 trial 的 motion energy 序列

[N_total, ~] = size(Conf);
fprintf('Loaded %d trials.\n', N_total);

%% 1. 重新算 residual volatility kernel -------------------------------
% 参数可以按你之前一贯用的
nTimeBins = 40;
winLen    = 10;

[resVol_mat, t_norm] = compute_resVol_kernel(motion_en, coh, nTimeBins, winLen);
fprintf('Residual volatility kernel computed: %d trials x %d time bins.\n', ...
        size(resVol_mat,1), size(resVol_mat,2));

%% (小 sanity check) 看一下一个被试的 resVol 分布
s_debug        = 1;
uniqSubj_all   = unique(subjID);          % 先把 unique(subjID) 存下来
thisSubj_debug = uniqSubj_all(s_debug);   % 选第一个被试的 ID

idx_dbg  = (subjID == thisSubj_debug);
V_dbg    = resVol_mat(idx_dbg, :);
std_each = std(V_dbg, 0, 1);

fprintf('Example subj: std(resVol) across time bins: [%.4f, %.4f]\n', ...
        min(std_each), max(std_each));

%% 2. 按 subject 做 confidence regression → biasKernel, sensKernel -----
uniqSubj  = unique(subjID);
nSubj     = numel(uniqSubj);
K         = nTimeBins;

biasKernel = nan(nSubj, K);   % β_vol(t)
sensKernel = nan(nSubj, K);   % δ_vol(t)
confBias   = nan(nSubj, 1);   % intercept β0
gamma      = nan(nSubj, 1);   % Correct 的总体效应 γ

for s = 1:nSubj
    thisSubj = uniqSubj(s);
    idx      = (subjID == thisSubj);

    y        = Conf(idx);           % Ns x 1
    cor      = Correct(idx);        % Ns x 1
    V_s      = resVol_mat(idx, :);  % Ns x K
    Ns       = numel(y);

    fprintf('Subject %d: %d trials.\n', thisSubj, Ns);

    % ---------- 2.1 baseline：只用 Correct 拿整体 conf bias ----------
    % logit P(Conf=1) = β0 + γ * Correct
    X0 = cor;  % glmfit 会自动加 intercept
    b0 = glmfit(X0, y, 'binomial', 'logit');
    confBias(s) = b0(1);   % 截距 = overall 高/低信心偏好
    gamma(s)    = b0(2);   % Correct 的整体效应

    % ---------- 2.2 time-bin 回归：Correct + V + V*Correct -----------
    for k = 1:K
        Vk = V_s(:, k);    % Ns x 1

        % 只要求 Vk 不是 NaN
        valid = ~isnan(Vk);
        yk    = y(valid);
        cor_k = cor(valid);
        Vk_k  = Vk(valid);

        % 这个 bin 有效 trial 太少 → 跳过
        if numel(yk) < 20
            biasKernel(s,k) = NaN;
            sensKernel(s,k) = NaN;
            continue;
        end

        % Vk 本身几乎没有变化 → 没法估系数
        if std(Vk_k) < 1e-6
            biasKernel(s,k) = NaN;
            sensKernel(s,k) = NaN;
            continue;
        end

        % 只有全对 or 全错：只能估 bias，没法估 sensitivity
        if numel(unique(cor_k)) == 1
            Vk_c = Vk_k - mean(Vk_k);

            % logit P(Conf=1) = β0 + β1*V
            Xk = Vk_c;  % 单一 predictor + intercept
            b  = glmfit(Xk, yk, 'binomial', 'logit');

            biasKernel(s,k) = b(2);   % V 的主效应
            sensKernel(s,k) = NaN;
        else
            % 对/错都有 → 可以估 bias + sensitivity
            cor_c = cor_k - mean(cor_k);
            Vk_c  = Vk_k  - mean(Vk_k);
            VxC   = Vk_c .* cor_c;

            % logit P(Conf=1) = β0 + β1*Correct + β2*V + β3*(V*Correct)
            Xk = [cor_c, Vk_c, VxC];
            b  = glmfit(Xk, yk, 'binomial', 'logit');

            biasKernel(s,k) = b(3);   % β2 → β_vol(t)
            sensKernel(s,k) = b(4);   % β3 → δ_vol(t)
        end
    end

    fprintf('   finite bias bins = %d, finite sens bins = %d\n', ...
        sum(isfinite(biasKernel(s,:))), sum(isfinite(sensKernel(s,:))));
end

%% 3. （可选）把 kernel 保存一下 --------------------------------------
save('conf_kernels_perSubj.mat', ...
     'biasKernel', 'sensKernel', 'confBias', 'gamma', ...
     't_norm', 'uniqSubj', 'nTimeBins', 'winLen');

fprintf('Kernels saved to conf_kernels_perSubj.mat\n');

%% 4. 用 kernel 预测每个 trial 的 p(high conf) ------------------------
% 我们先看一个被试（你可以改成一个 for-loop，看所有人）

s        = 1;                 % 第一个被试
thisSubj = uniqSubj(s);
idx_s    = (subjID == thisSubj);

V_s    = resVol_mat(idx_s, :);   % Ns x K
cor_s  = Correct(idx_s);         % Ns x 1
conf_s = Conf(idx_s);            % Ns x 1
Ns     = numel(conf_s);
K      = nTimeBins;

% 取出这个被试的 kernel & baseline 参数
beta0      = confBias(s);       % 截距 β0
gamma_s    = gamma(s);          % Correct 的整体效应 γ
beta_vol   = biasKernel(s, :);  % 1 x K
delta_vol  = sensKernel(s, :);  % 1 x K

% ---- 4.1 计算每个 trial 的 logit & p(high conf) ---------------------
% volatility 对 bias 的总贡献：sum_k β_vol(tk) * V_j(tk)
vol_bias_term = V_s * beta_vol.';             % Ns x 1

% volatility 对 sensitivity 的总贡献：sum_k δ_vol(tk) * V_j(tk) * Correct_j
vol_sens_term = (V_s .* cor_s) * delta_vol.'; % Ns x 1

% baseline performance 项：γ * Correct_j
base_perf_term = gamma_s * cor_s;             % Ns x 1

% logit
eta = beta0 + base_perf_term + vol_bias_term + vol_sens_term;  % Ns x 1

% logistic → p(high)
pHigh_pred = 1 ./ (1 + exp(-eta));            % Ns x 1

% 和真实 high/low 做相关，看看预测效果
[r_all, p_all] = corr(pHigh_pred, conf_s);
fprintf('\nSubj %d: corr(predicted pHigh, actual high/low) = %.3f (p = %.3g)\n', ...
        thisSubj, r_all, p_all);

%% 5. 画图：trial-by-trial prediction & build-up over time ------------
figure('Name', sprintf('Subject %d volatility-based prediction', thisSubj));

% 5.1 左图：pred p(high) vs actual high/low
subplot(1,2,1);
scatter(pHigh_pred, conf_s + 0.02*randn(size(conf_s)), 5, 'filled');
xlabel('predicted p(high)');
ylabel('actual high (=1) / low (=0)');
title(sprintf('Subj %d: trial-by-trial prediction (r = %.2f)', thisSubj, r_all));
xlim([0 1]); ylim([-0.2 1.2]); grid on;

% 5.2 右图：随着加入更多 time bin，预测能力如何提升（build-up）
corr_vs_T = nan(1, K);

for T = 1:K
    beta_T  = beta_vol(1:T);
    delta_T = delta_vol(1:T);

    vol_bias_T = V_s(:,1:T) * beta_T.';                  % 前 T 个 bin 的 bias 部分
    vol_sens_T = (V_s(:,1:T) .* cor_s) * delta_T.';      % 前 T 个 bin 的 sensitivity 部分

    eta_T   = beta0 + gamma_s*cor_s + vol_bias_T + vol_sens_T;
    pHigh_T = 1 ./ (1 + exp(-eta_T));

    corr_vs_T(T) = corr(pHigh_T, conf_s);
end

subplot(1,2,2);
plot(t_norm, corr_vs_T, '-o');
xlabel('Normalized time (0–1)');
ylabel('corr(predicted pHigh, actual)');
title(sprintf('Subj %d: build-up of volatility-based prediction', thisSubj));
grid on;
