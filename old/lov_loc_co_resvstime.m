%% plot_resVol_lowCoh_lowVol_correct.m
% 目标：
% 在 low coherence + low volatility + correct trials 的条件下，
% 画出 residual volatility 随 normalized time (0–1) 变化的一条平均曲线。

clear; clc;

%% 0. 读入数据
data_path = 'all_with_me.mat';   % 改成你的文件路径
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell，每个是 nFrames x 1 double
coh          = allStruct.rdm1_coh(:);      % coherence per trial
coh_std      = allStruct.rdm1_coh_std(:);  % coherence std = volatility level
correct      = allStruct.correct(:);       % 0/1，是否答对

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. 定义 low coherence & low volatility

% ---- low coherence：所有非 0 coherence 的绝对值最小值 ----
coh_abs_0 = unique(abs(coh(coh == 0)));
coh_low      = min(coh_abs_0);
fprintf('Low coherence level (abs): %.4f\n', coh_low);

idx_lowCoh = (abs(coh) == coh_low);

% ---- low volatility：coh_std 在非 0 coherence 里的最小值 ----
std_vals = unique(coh_std(coh ~= 0));
std_low  = min(std_vals);
fprintf('Low volatility std level: %.4f\n', std_low);

idx_lowVol = (coh_std == std_low);

% ---- correct trials ----
idx_correct = (correct == 1);

%% 2. 对每个 trial 做 sliding window：算 mean evidence & std

winLen = 10;   % window 长度（frame 数）

mean_win = cell(nTrials, 1);  % 每个 trial：窗口序列的 mean evidence
std_win  = cell(nTrials, 1);  % 每个 trial：窗口序列的 std（volatility）

for tr = 1:nTrials
    frames = motion_energy{tr};    % nFrames x 1 double
    if isempty(frames)
        mean_win{tr} = [];
        std_win{tr}  = [];
        continue;
    end
    
    trace = frames(:)';            % 1 x nFrames
    
    % 去掉结尾 padding 的 0
    tol    = 1e-12;
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        mean_win{tr} = [];
        std_win{tr}  = [];
        continue;
    end
    trace = trace(1:last_nz);
    
    nF = numel(trace);
    if nF < winLen
        mean_win{tr} = [];
        std_win{tr}  = [];
        continue;
    end
    
    nWin = nF - winLen + 1;
    mw   = nan(1, nWin);
    sw   = nan(1, nWin);
    
    for w = 1:nWin
        seg = trace(w : w+winLen-1);
        mw(w) = mean(seg);
        sw(w) = std(seg, 0);
    end
    
    mean_win{tr} = mw;
    std_win{tr}  = sw;
end

%% 3. 把 mean & std 压到 0–1 时间轴上的固定 bin（比如 40 个）

nBins = 40;
t_norm = linspace(0, 1, nBins);    % 0–1 normalized time

MEAN_norm = nan(nTrials, nBins);
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mw = mean_win{tr};
    sw = std_win{tr};
    
    if isempty(mw) || numel(mw) < 2
        continue;
    end
    
    nW     = numel(mw);
    t_orig = linspace(0, 1, nW);   % 原始窗口的 0–1 时间坐标
    
    MEAN_norm(tr, :) = interp1(t_orig, mw, t_norm, 'linear', 'extrap');
    STD_norm(tr, :)  = interp1(t_orig, sw, t_norm, 'linear', 'extrap');
end

%% 4. 在每个时间点做回归：STD ~ |MEAN| + coherence，拿 residual STD

resid_STD = nan(size(STD_norm));   % nTrials x nBins

for b = 1:nBins
    y = STD_norm(:, b);            % 当前时间点的 std（所有 trial）
    
    valid = ~isnan(y);
    if sum(valid) < 10
        continue;
    end
    
    yv = y(valid);
    mv = MEAN_norm(valid, b);
    cv = coh(valid);              % coherence
    
    % 设计矩阵：intercept, |mean|, coherence
    X = [ones(sum(valid), 1), abs(mv), cv];
    
    % 线性回归： y = X * beta + residual
    beta  = X \ yv;
    y_hat = X * beta;
    r     = yv - y_hat;           % residual std
    
    resid_STD(valid, b) = r;
end

%% 5. 选出 low coh + low vol + correct 的 trial

idx_use = idx_lowCoh & idx_lowVol & idx_correct;
fprintf('Trials with low coh + low vol + correct: %d\n', sum(idx_use));

resid_sub = resid_STD(idx_use, :);

%% 6. 对这些 trial 求时间上的平均 residual volatility，并平滑，然后画图

mean_resid        = mean(resid_sub, 1, 'omitnan');      % 1 x nBins
mean_resid_smooth = smoothdata(mean_resid, 'gaussian', 5);

figure;
plot(t_norm, mean_resid_smooth, 'LineWidth', 2);
xlabel('Normalized time (0–1)');
ylabel('Residual volatility');
title('Low coherence & Low volatility & Correct trials');
grid on;
