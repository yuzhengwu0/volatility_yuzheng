%% logistic_vo_model.m
% 用 Early/Mid/Late 的 mean evidence + residual volatility 预测 correct (0/1)

clear; clc;

% ===== 1. 从文件里读数据（如果你已经在 workspace 有这些变量，可以把这块注释掉） =====
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

% 这里假设你已经另外算好 MEAN_norm 和 resid_STD
% 如果已经在 workspace 里，就把下面两行注释掉或换成你自己的 mat 文件
load('Volatility_window_evidence.m', 'earlymidlate_vo_pre_model.m', 'volatility_residual.mat.m');

correct_vec = allStruct.correct(:);   % 0 = incorrect, 1 = correct

% ===== 2. 定义 Early / Mid / Late 的时间段 =====
[nTrials, nBins] = size(MEAN_norm);

% 你之前用的是 40 个 bin，这里按同样方式切三段
early_idx = 1:13;          % 0   ~ 0.33
mid_idx   = 14:26;         % 0.33~ 0.66
late_idx  = 27:nBins;      % 0.66~ 1

% ===== 3. 对每个 trial 求 6 个 summary predictor =====
% 3 个 mean evidence
mean_E_mean = mean(MEAN_norm(:, early_idx), 2, 'omitnan');
mean_M_mean = mean(MEAN_norm(:, mid_idx),   2, 'omitnan');
mean_L_mean = mean(MEAN_norm(:, late_idx),  2, 'omitnan');

% 3 个 residual volatility
mean_E_resVol = mean(resid_STD(:, early_idx), 2, 'omitnan');
mean_M_resVol = mean(resid_STD(:, mid_idx),   2, 'omitnan');
mean_L_resVol = mean(resid_STD(:, late_idx),  2, 'omitnan');

% 设计矩阵：每一列就是一个 predictor
% 列顺序：Early mean, Mid mean, Late mean, Early resVol, Mid resVol, Late resVol
X = [mean_E_mean, mean_M_mean, mean_L_mean, ...
     mean_E_resVol, mean_M_resVol, mean_L_resVol];

y = correct_vec;   % 0/1

% ===== 4. 去掉含 NaN 的 trial =====
valid_mask = all(~isnan(X), 2) & ~isnan(y);
X_valid    = X(valid_mask, :);
y_valid    = y(valid_mask);

fprintf('Using %d valid trials for logistic regression.\n', sum(valid_mask));

% ===== 5. Logistic 回归：correct ~ 6 个 predictor =====
% glmfit 会自动加一个截距项，所以不用在 X 里自己加 1 列
[b, dev, stats] = glmfit(X_valid, y_valid, 'binomial', 'link', 'logit');

% b 的含义：
% b(1)  截距 Intercept
% b(2)  Early mean evidence
% b(3)  Mid   mean evidence
% b(4)  Late  mean evidence
% b(5)  Early residual volatility
% b(6)  Mid   residual volatility
% b(7)  Late  residual volatility

predictor_names = { ...
    'Intercept', ...
    'Mean_Early', ...
    'Mean_Mid', ...
    'Mean_Late', ...
    'ResVol_Early', ...
    'ResVol_Mid', ...
    'ResVol_Late'};

fprintf('\nLogistic regression results (predicting correct = 1):\n');
for k = 1:numel(predictor_names)
    fprintf('%15s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        predictor_names{k}, b(k), stats.se(k), stats.p(k));
end
