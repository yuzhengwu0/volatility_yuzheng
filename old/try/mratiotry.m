%% window_volatility_Mratio.m
% 对每一个时间 bin，看 high vs low residual volatility 的 M-ratio 差别

clear; clc;

%% 1. 读数据：行为 + residual volatility

data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

correct_vec = allStruct.correct(:);      % 0/1
resp_vec    = allStruct.req_resp(:);     % 1=right, 2=left
conf_vec    = allStruct.confidence(:);         % 假设是 1..K 的 rating

% 恢复 stimulus：1=S1, 2=S2
stim_vec = resp_vec;
stim_vec(correct_vec == 0) = 3 - resp_vec(correct_vec == 0);

% 载入你之前算好的 residual volatility (trial x timeBin)
% 这里假设文件里叫 resid_STD，维度是 nTrials x nBins
load('volatility_residual.mat', 'resid_STD');  

[nTrials, nBins] = size(resid_STD);
fprintf('Trials = %d, time bins = %d\n', nTrials, nBins);

%% 2. 准备：一些有效 trial 的 mask（conf/stim/resp 都不是 NaN）

valid_basic = ~isnan(correct_vec) & ~isnan(resp_vec) & ~isnan(conf_vec);

% confidence 等级数（比如 6 级）
K = max(conf_vec);

% 结果存这里：每个时间 bin 的 low / high volatility M-ratio
Mratio_low  = nan(1, nBins);
Mratio_high = nan(1, nBins);

for b = 1:nBins
    
    v_b = resid_STD(:, b);     % 当前时间 bin 的 residual volatility
    mask_b = valid_basic & ~isnan(v_b);
    
    if sum(mask_b) < 50
        % 有效 trial 太少就略过
        fprintf('Bin %d: 有效 trial 太少，跳过。\n', b);
        continue;
    end
    
    v_use = v_b(mask_b);
    
    % median split：低 volatility vs 高 volatility
    med_v = median(v_use);
    low_mask_global  = (v_b <= med_v);
    high_mask_global = (v_b >  med_v);
    
    idx_low  = find(mask_b & low_mask_global);
    idx_high = find(mask_b & high_mask_global);
    
    if numel(idx_low) < 30 || numel(idx_high) < 30
        fprintf('Bin %d: low/high 组 trial 都要 >=30，这里太少，跳过。\n', b);
        continue;
    end
    
    % ==== 对低 volatility 组算 M-ratio ====
    Mratio_low(b) = compute_Mratio_subset(idx_low, stim_vec, resp_vec, conf_vec, K);
    
    % ==== 对高 volatility 组算 M-ratio ====
    Mratio_high(b) = compute_Mratio_subset(idx_high, stim_vec, resp_vec, conf_vec, K);
    
    fprintf('Bin %2d: n_low=%d, n_high=%d, M_low=%.2f, M_high=%.2f\n', ...
        b, numel(idx_low), numel(idx_high), Mratio_low(b), Mratio_high(b));
end

%% 3. 画图：M-ratio 随时间的变化（高vs低 volatility）

t_norm = linspace(0, 1, nBins);   % 0=trial开始, 1=trial结束

figure;
plot(t_norm, Mratio_low,  '-o'); hold on;
plot(t_norm, Mratio_high, '-o');
yline(1, '--');   % 理论完美 = 1
xlabel('Normalized time within trial (0=start, 1=end)');
ylabel('M-ratio (meta-d'' / d'')');
legend({'Low residual volatility', 'High residual volatility', 'M-ratio = 1'});
title('Effect of window residual volatility on M-ratio over time');
grid on;

function Mratio = compute_Mratio_subset(idx, stim_vec, resp_vec, conf_vec, K)
% idx: 使用哪些 trial
% stim_vec: 1/2 (S1/S2)
% resp_vec: 1/2 (report S1/S2)
% conf_vec: 1..K
% K: rating 级数
%
% 返回 M-ratio（如果 fit 失败就 NaN）

    stim = stim_vec(idx);
    resp = resp_vec(idx);
    conf = conf_vec(idx);

    % 先把每个 trial 的 S/resp/conf 拿出来
    % 按 Maniscalco & Lau toolbox 的约定构造 nR_S1, nR_S2
    %
    % 这里假设：nR_S1 和 nR_S2 的长度都是 2K：
    % nR_S1 = [Resp=S1, rating=K..1,  Resp=S2, rating=1..K]
    % nR_S2 = [Resp=S1, rating=K..1,  Resp=S2, rating=1..K]
    % *** 请用你本地 fit_meta_d_MLE.m 顶部的说明确认一下顺序，如果不一样，改这里 ***

    nR_S1 = zeros(1, 2*K);
    nR_S2 = zeros(1, 2*K);

    for r = 1:K
        % ----- Stimulus S1 -----
        % S1, response = S1, conf = r
        nR_S1(K+1-r) = sum(stim==1 & resp==1 & conf==r);   % 高信心放前面(K→1)

        % S1, response = S2, conf = r
        nR_S1(K + r) = sum(stim==1 & resp==2 & conf==r);   % 从左到右 1..K

        % ----- Stimulus S2 -----
        nR_S2(K+1-r) = sum(stim==2 & resp==1 & conf==r);
        nR_S2(K + r) = sum(stim==2 & resp==2 & conf==r);
    end

    % 调用 meta-d' toolbox 拟合
    try
        fit = fit_meta_d_MLE(nR_S1, nR_S2);
        Mratio = fit.meta_d ./ fit.d;
    catch ME
        warning('meta-d'' fit failed in this subset: %s', ME.message);
        Mratio = NaN;
    end
end

