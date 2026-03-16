%% logit_cumulative_evidence.m
% 目标：
% 1) 从 all.motion_energy 读出 frame-by-frame signed evidence（右正左负）
% 2) 对每个 trial 做“从头到尾的累积和” → 决策变量轨迹 DV(t)
% 3) 把每个 trial 的 DV(t) 归一化到 0–1 时间轴（统一 40 个时间点）
% 4) 把时间切成 early / mid / late 三段，取在每段结束时的累积 evidence
% 5) 用 logistic regression 看 Early/Mid/Late 的累积 evidence 能否预测 correct（0/1）

clear; clc;

%% 0. 读入数据
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell（差值 evidence，右正左负）
correct_vec   = allStruct.correct(:);      % 1 = correct, 0 = incorrect
nTrials       = numel(motion_energy);

fprintf('Loaded %d trials.\n', nTrials);

%% 1. 对每个 trial 计算“累积 evidence”（去掉尾巴占位 0）

tol = 1e-12;
cum_evidence = cell(nTrials, 1);   % 每个 trial 一条 DV(t) 轨迹（累积和）

for tr = 1:nTrials
    frames = motion_energy{tr};   % nFrames x 1 double
    trace  = frames(:)';          % 1 x nFrames
    
    % 去掉末尾 padding 的 0：只保留到最后一个非 0 帧
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        cum_evidence{tr} = [];
        continue;
    end
    
    trace_eff = trace(1:last_nz);        % 有效部分
    dv_tr     = cumsum(trace_eff);       % 累积 evidence（决策变量轨迹）
    
    cum_evidence{tr} = dv_tr;
end

%% 2. 把每个 trial 的 DV(t) 插值到统一的 0–1 时间轴（40 个时间点）

nBins  = 40;                       % 想要的归一化时间点数
t_norm = linspace(0, 1, nBins);    % 0 = trial 开始, 1 = trial 结束

CUM_norm = nan(nTrials, nBins);    % trial x time

for tr = 1:nTrials
    dv_tr = cum_evidence{tr};      % 这个 trial 的累积 evidence 轨迹
    
    if isempty(dv_tr)
        continue;
    end
    
    nFrames_eff = numel(dv_tr);
    t_orig      = linspace(0, 1, nFrames_eff);   % 这个 trial 自己的 0–1 时间
    
    % 插值到统一的 t_norm
    CUM_norm(tr, :) = interp1(t_orig, dv_tr, t_norm, 'linear');
end

%% 3. 定义 Early / Mid / Late 段，并取“该段结束时”的累积 evidence

% 按 40 个 bin 划分三段：
early_idx = 1:13;          % 大约 0 ~ 0.33
mid_idx   = 14:26;         % 大约 0.33 ~ 0.66
late_idx  = 27:40;         % 大约 0.66 ~ 1

% 这里我们取“每一段结束时”的累积 evidence：
%   Early：在 early 段最后一个时间点的 DV
%   Mid  ：在 mid 段最后一个时间点的 DV
%   Late ：在 late 段最后一个时间点的 DV
cum_E = CUM_norm(:, early_idx(end));   % trial x 1
cum_M = CUM_norm(:, mid_idx(end));     % trial x 1
cum_L = CUM_norm(:, late_idx(end));    % trial x 1

% 如果你更想用“该段内的平均 DV”，可以改成：
% cum_E = mean(CUM_norm(:, early_idx), 2, 'omitnan');
% cum_M = mean(CUM_norm(:, mid_idx),   2, 'omitnan');
% cum_L = mean(CUM_norm(:, late_idx),  2, 'omitnan');

%% 4. 组装设计矩阵并做 logistic 回归：correct ~ Cum_E + Cum_M + Cum_L

X = [cum_E, cum_M, cum_L];    % 每列一个 predictor
y = correct_vec;              % 0/1

% 去掉含 NaN 的 trial
valid_mask = all(~isnan(X), 2) & ~isnan(y);
X_valid    = X(valid_mask, :);
y_valid    = y(valid_mask);

fprintf('Using %d valid trials for logistic regression.\n', sum(valid_mask));

if isempty(X_valid)
    error('No valid trials left after removing NaNs.');
end

% glmfit 会自动加一个截距列，所以 X_valid 不用自己加全 1
[b, dev, stats] = glmfit(X_valid, y_valid, 'binomial', 'link', 'logit');

% b 的含义：
% b(1) 截距 Intercept
% b(2) 累积 evidence at end of Early 段
% b(3) 累积 evidence at end of Mid   段
% b(4) 累积 evidence at end of Late  段

predictor_names = { ...
    'Intercept', ...
    'Cum_Early', ...
    'Cum_Mid', ...
    'Cum_Late'};

fprintf('\nLogistic regression results (predicting correct = 1):\n');
for k = 1:numel(predictor_names)
    fprintf('%10s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        predictor_names{k}, b(k), stats.se(k), stats.p(k));
end

%% 5. 可选：画一下“正确 vs 错误”的平均 cumulative evidence 轨迹（方便直觉理解）

idxC = (correct_vec == 1);
idxI = (correct_vec == 0);

mean_CUM_C = mean(CUM_norm(idxC, :), 1, 'omitnan');
mean_CUM_I = mean(CUM_norm(idxI, :), 1, 'omitnan');

figure;
subplot(2,1,1);
plot(t_norm, mean_CUM_C, 'g-', 'LineWidth', 1.5); hold on;
plot(t_norm, mean_CUM_I, 'm-', 'LineWidth', 1.5);
xlabel('Normalized time within trial (0 = start, 1 = end)');
ylabel('Cumulative evidence');
legend({'Correct','Incorrect'});
title('Normalized-time cumulative evidence (Correct vs Incorrect)');
grid on;

subplot(2,1,2);
plot(t_norm, mean_CUM_C - mean_CUM_I, 'k-', 'LineWidth', 1.5);
xlabel('Normalized time within trial (0 = start, 1 = end)');
ylabel('Evidence difference (Correct - Incorrect)');
title('Cumulative evidence accuracy kernel');
grid on;
