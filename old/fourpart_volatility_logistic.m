%% logit_mean_resVol_correct_Q4.m
% 目标：
% 1) 从 all.motion_energy 重新算每个 trial 的 sliding-window mean & std
% 2) 归一化到 0–1 时间轴（40 个 time bin）
% 3) 在每个时间点上，用 STD ~ |MEAN| 回归，拿 residual STD（额外 volatility）
% 4) 把时间按 0.25 切成 4 段 (Q1–Q4)，分别取 mean evidence & residual volatility
% 5) 用 logistic regression 看这些 predictor 能否预测 correct（0/1）

clear; clc;

%% 0. 读入数据
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell（差值 evidence，右正左负）
correct_vec   = allStruct.correct(:);      % 1 = correct, 0 = incorrect
nTrials       = numel(motion_energy);

fprintf('Loaded %d trials.\n', nTrials);

%% 1. 对每个 trial 做 sliding window，算 mean & std（去尾巴 0）

winLen = 10;
tol    = 1e-12;

evidence_strength   = cell(nTrials, 1);   % 每个 trial: window mean
volatility_strength = cell(nTrials, 1);   % 每个 trial: window std

for tr = 1:nTrials
    frames = motion_energy{tr};      % nFrames x 1
    trace  = frames(:)';             % 1 x nFrames（可能带 padding 0）
    
    % 去掉结尾占位 0（只保留到最后一个非 0）
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

%% 2. 把 mean & std 都插值到统一的 0–1 时间轴

nBins  = 40;                          % 归一化时间上的 bin 数
t_norm = linspace(0, 1, nBins);       % 0 = trial 开始，1 = 结束

MEAN_norm = nan(nTrials, nBins);      % trial x time
STD_norm  = nan(nTrials, nBins);      % trial x time

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};    % 这个 trial 的 mean evidence
    sd_tr = volatility_strength{tr};  % 这个 trial 的 volatility
    
    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end
    
    % 正常情况下长度应该一样，这里保险起见取较短那一段
    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);
    
    t_orig = linspace(0, 1, nWin_tr);   % 这个 trial 自己的 0–1 进度
    
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
        continue;                 % 有效 trial 太少就跳过
    end
    
    Xlin  = [ones(sum(mask),1), x(mask)];   % [常数项, |mean|]
    beta  = Xlin \ y(mask);                 % 最小二乘拟合直线 y = b0 + b1*x
    y_hat = Xlin * beta;                    % 在这个时间点上“正常”应该有的 std
    
    tmp       = nan(size(y));
    tmp(mask) = y(mask) - y_hat;            % residual = 真实 std - 正常 std
    resid_STD(:, b) = tmp;                  % 这就是“额外 volatility”
end

%% 4. 把时间按 0.25 切成四段 Q1 / Q2 / Q3 / Q4，取 mean & residual volatility

% 这里 nBins = 40，所以每段 10 个 bin
qSize = nBins / 4;   % = 10
Q1_idx = 1:qSize;                          % 0   ~ 0.25
Q2_idx = (qSize+1):(2*qSize);             % 0.25~ 0.50
Q3_idx = (2*qSize+1):(3*qSize);           % 0.50~ 0.75
Q4_idx = (3*qSize+1):(4*qSize);           % 0.75~ 1.00

% 为每个 trial 生成 8 个 predictor：
% 四段的 mean evidence + 四段的 residual volatility

mean_Q1_mean = mean(MEAN_norm(:, Q1_idx), 2, 'omitnan');
mean_Q2_mean = mean(MEAN_norm(:, Q2_idx), 2, 'omitnan');
mean_Q3_mean = mean(MEAN_norm(:, Q3_idx), 2, 'omitnan');
mean_Q4_mean = mean(MEAN_norm(:, Q4_idx), 2, 'omitnan');

mean_Q1_resVol = mean(resid_STD(:, Q1_idx), 2, 'omitnan');
mean_Q2_resVol = mean(resid_STD(:, Q2_idx), 2, 'omitnan');
mean_Q3_resVol = mean(resid_STD(:, Q3_idx), 2, 'omitnan');
mean_Q4_resVol = mean(resid_STD(:, Q4_idx), 2, 'omitnan');

% 组装成设计矩阵 X
X = [mean_Q1_mean, mean_Q2_mean, mean_Q3_mean, mean_Q4_mean, ...
     mean_Q1_resVol, mean_Q2_resVol, mean_Q3_resVol, mean_Q4_resVol];

y = correct_vec;   % 0/1（0 = incorrect, 1 = correct）

% 去掉有 NaN 的 trial
valid_mask = all(~isnan(X), 2) & ~isnan(y);
X_valid    = X(valid_mask, :);
y_valid    = y(valid_mask);

fprintf('Using %d valid trials for logistic regression.\n', sum(valid_mask));

%% 5. 做 logistic 回归：correct ~ mean(Q1–Q4) + residualVol(Q1–Q4)

if isempty(X_valid)
    error('No valid trials left after removing NaNs.');
end

% glmfit 默认会加一个常数项，所以这里的 X_valid 不需要再加全 1 列
[b, dev, stats] = glmfit(X_valid, y_valid, 'binomial', 'link', 'logit');

% b 的含义：
% b(1): 截距
% b(2): Q1 mean evidence
% b(3): Q2 mean evidence
% b(4): Q3 mean evidence
% b(5): Q4 mean evidence
% b(6): Q1 residual volatility
% b(7): Q2 residual volatility
% b(8): Q3 residual volatility
% b(9): Q4 residual volatility

predictor_names = { ...
    'Intercept', ...
    'Mean_Q1', ...
    'Mean_Q2', ...
    'Mean_Q3', ...
    'Mean_Q4', ...
    'ResVol_Q1', ...
    'ResVol_Q2', ...
    'ResVol_Q3', ...
    'ResVol_Q4'};

fprintf('\nLogistic regression results (predicting correct = 1):\n');
for k = 1:numel(predictor_names)
    fprintf('%12s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        predictor_names{k}, b(k), stats.se(k), stats.p(k));
end

%% 6. 可选：画一下 Q1–Q4 residual volatility 在 correct vs incorrect 的分布

idxC = (correct_vec == 1) & valid_mask;
idxI = (correct_vec == 0) & valid_mask;

figure;
subplot(4,1,1);
histogram(mean_Q1_resVol(idxC), 'Normalization','pdf'); hold on;
histogram(mean_Q1_resVol(idxI), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Residual volatility (Q1, 0–0.25)');
ylabel('Density');
title('Q1 residual volatility distribution');
grid on;

subplot(4,1,2);
histogram(mean_Q2_resVol(idxC), 'Normalization','pdf'); hold on;
histogram(mean_Q2_resVol(idxI), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Residual volatility (Q2, 0.25–0.5)');
ylabel('Density');
title('Q2 residual volatility distribution');
grid on;

subplot(4,1,3);
histogram(mean_Q3_resVol(idxC), 'Normalization','pdf'); hold on;
histogram(mean_Q3_resVol(idxI), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Residual volatility (Q3, 0.5–0.75)');
ylabel('Density');
title('Q3 residual volatility distribution');
grid on;

subplot(4,1,4);
histogram(mean_Q4_resVol(idxC), 'Normalization','pdf'); hold on;
histogram(mean_Q4_resVol(idxI), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Residual volatility (Q4, 0.75–1)');
ylabel('Density');
title('Q4 residual volatility distribution');
grid on;
