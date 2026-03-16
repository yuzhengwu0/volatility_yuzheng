%% cumu_noise_by_coh.m
% 1) 从 all.motion_energy 读入 trial 级 evidence (right - left)
% 2) 按 coherence 分成 low / high 两组
% 3) 分别为每组算 cumulative evidence & cumulative noise（running variance）
% 4) 把 trace 归一化到 [0,1] 时间轴
% 5) 把时间分成 Early / Mid / Late 三段，做 logistic 回归预测 correct

clear; clc;

%% 0. 读入数据
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell (right-left evidence)
correct_vec   = allStruct.correct(:);      % 1 = correct, 0 = incorrect

% ---- 这里用 coherence 字段分 high / low ----
% 如果字段名不是 coherence，就改成你自己的
coh = allStruct.rdm1_coh(:);             % e.g. 0.05, 0.1, 0.2, 0.4

% 一个简单的分法：0.05 / 0.1 视为 low，0.2 / 0.4 视为 high
low_mask  = (coh <= 100);
high_mask = (coh >= 101);

idx_low   = find(low_mask);
idx_high  = find(high_mask);

fprintf('Total trials: %d\n', numel(motion_energy));
fprintf('Low coherence trials:  %d\n', numel(idx_low));
fprintf('High coherence trials: %d\n', numel(idx_high));

% 归一化时间轴设置
nBins  = 40;
t_norm = linspace(0, 1, nBins);   % 0 = trial 开始，1 = 结束

% 时间段切分：Early / Mid / Late
early_idx = 1:13;     % 0 ~ 0.25 左右
mid_idx   = 14:26;    % 0.25 ~ 0.5 左右
late_idx  = 27:40;    % 0.5 ~ 1

%% 1. 在 low coherence trial 上做分析
fprintf('\n=== LOW coherence trials ===\n');
run_cumu_noise_block(motion_energy, correct_vec, idx_low, ...
                     t_norm, early_idx, mid_idx, late_idx, ...
                     'LOW coherence');

%% 2. 在 high coherence trial 上做分析
fprintf('\n=== HIGH coherence trials ===\n');
run_cumu_noise_block(motion_energy, correct_vec, idx_high, ...
                     t_norm, early_idx, mid_idx, late_idx, ...
                     'HIGH coherence');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 局部函数：对一组 trial index 做 cumulative noise 分析
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_cumu_noise_block(motion_energy, correct_vec, trial_idx, ...
                              t_norm, early_idx, mid_idx, late_idx, label_str)

nBins = numel(t_norm);
tol   = 1e-12;

nSubTrials = numel(trial_idx);

% 保存每个 trial 在归一化时间上的 cumulative evidence & noise
CumEv_norm    = nan(nSubTrials, nBins);   % trial x time
CumNoise_norm = nan(nSubTrials, nBins);   % trial x time
y_correct     = nan(nSubTrials, 1);       % 该 trial 是否答对

for k = 1:nSubTrials
    tr    = trial_idx(k);
    frames = motion_energy{tr};       % nFrames x 1
    trace  = frames(:)';              % 1 x nFrames（可能带 0 padding）
    
    % 去掉结尾占位 0
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        continue;
    end
    
    trace_eff = trace(1:last_nz);     % 有效部分
    nFrames   = numel(trace_eff);
    
    % ---- 累积 evidence & 累积 noise（running variance）----
    % 累积和 & 累积平方和
    cum_sum = cumsum(trace_eff);
    cum_sq  = cumsum(trace_eff.^2);
    n_vec   = 1:nFrames;
    
    % running mean = 当前为止的平均证据
    cum_ev = cum_sum ./ n_vec;
    
    % running variance (噪音强度) 这里用有偏估计 n 分母就够
    cum_var = cum_sq ./ n_vec - (cum_sum ./ n_vec).^2;
    
    % 把每个 trial 的 trace 插值到统一 0–1 时间轴
    t_orig = linspace(0, 1, nFrames);
    CumEv_norm(k, :)    = interp1(t_orig, cum_ev,  t_norm, 'linear');
    CumNoise_norm(k, :) = interp1(t_orig, cum_var, t_norm, 'linear');
    
    % 记录这个 trial 是否答对
    y_correct(k) = correct_vec(tr);
end

% 去掉全 NaN 的 trial
valid_row = ~isnan(y_correct) & any(~isnan(CumEv_norm), 2);
CumEv_norm    = CumEv_norm(valid_row, :);
CumNoise_norm = CumNoise_norm(valid_row, :);
y_correct     = y_correct(valid_row);

fprintf('Using %d valid trials in this block (%s).\n', sum(valid_row), label_str);

%% 把 0–1 时间轴上数据分成 Early / Mid / Late，取平均做 predictor

CumEv_Early    = mean(CumEv_norm(:, early_idx),    2, 'omitnan');
CumEv_Mid      = mean(CumEv_norm(:, mid_idx),      2, 'omitnan');
CumEv_Late     = mean(CumEv_norm(:, late_idx),     2, 'omitnan');

CumNoise_Early = mean(CumNoise_norm(:, early_idx), 2, 'omitnan');
CumNoise_Mid   = mean(CumNoise_norm(:, mid_idx),   2, 'omitnan');
CumNoise_Late  = mean(CumNoise_norm(:, late_idx),  2, 'omitnan');

% 设计矩阵：Early / Mid / Late 的 cumulative evidence + cumulative noise
X = [CumEv_Early, CumEv_Mid, CumEv_Late, ...
     CumNoise_Early, CumNoise_Mid, CumNoise_Late];

y = y_correct;    % 0/1

valid_mask = all(~isnan(X), 2) & ~isnan(y);
X_valid    = X(valid_mask, :);
y_valid    = y(valid_mask);

fprintf('Logistic regression with %d trials (%s).\n', sum(valid_mask), label_str);

if isempty(X_valid)
    warning('No valid trials for glmfit in block "%s".', label_str);
    return;
end

% logistic 回归：correct ~ cumEv(E/M/L) + cumNoise(E/M/L)
[b, dev, stats] = glmfit(X_valid, y_valid, 'binomial', 'link', 'logit');

pred_names = { ...
    'Intercept', ...
    'CumEv_Early', 'CumEv_Mid', 'CumEv_Late', ...
    'CumNoise_Early', 'CumNoise_Mid', 'CumNoise_Late'};

fprintf('\nLogistic regression results (predicting correct = 1), block = %s:\n', label_str);
for i = 1:numel(pred_names)
    fprintf('%16s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        pred_names{i}, b(i), stats.se(i), stats.p(i));
end

%% 画直方图：看 cumulative noise 在 correct / incorrect 里的分布（E/M/L）

idxC = (y_correct == 1) & valid_mask;
idxI = (y_correct == 0) & valid_mask;

figure('Name', ['Cumulative noise vs correctness - ' label_str]);

subplot(3,1,1);
histogram(CumNoise_Early(idxC), 'Normalization','pdf'); hold on;
histogram(CumNoise_Early(idxI), 'Normalization','pdf');
xlabel('Cumulative noise (Early)');
ylabel('Density');
legend({'Correct','Incorrect'});
title(['Early cumulative noise vs correctness (' label_str ')']);
grid on;

subplot(3,1,2);
histogram(CumNoise_Mid(idxC), 'Normalization','pdf'); hold on;
histogram(CumNoise_Mid(idxI), 'Normalization','pdf');
xlabel('Cumulative noise (Mid)');
ylabel('Density');
legend({'Correct','Incorrect'});
title(['Mid cumulative noise vs correctness (' label_str ')']);
grid on;

subplot(3,1,3);
histogram(CumNoise_Late(idxC), 'Normalization','pdf'); hold on;
histogram(CumNoise_Late(idxI), 'Normalization','pdf');
xlabel('Cumulative noise (Late)');
ylabel('Density');
legend({'Correct','Incorrect'});
title(['Late cumulative noise vs correctness (' label_str ')']);
grid on;

end
