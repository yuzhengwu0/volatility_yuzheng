%% resVol_timecourse_lowHighVol_coh0.m
% 目标：
% 1) 从 all.motion_energy 做 sliding window，算 mean & std（volatility）
% 2) 把 mean/std 都 warp 到 0–1 时间轴（比如 40 个 bin）
% 3) 在每个时间点上，用 STD ~ |MEAN| + coherence 回归，取 residual STD = extra noise
% 4) 只挑 coherence == 0 的 trial
% 5) 在 coh == 0 里面再分 low vol vs high vol
% 6) 画一张图：横轴 time bin，纵轴 mean residual volatility
%    两条线：low vol & coh==0；high vol & coh==0

clear; clc;

%% 0. 读入数据
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;     % 7584 x 1 cell（右正左负）
cohVec        = allStruct.rdm1_coh(:);      % coherence per trial  (根据你自己的字段改)
volVec        = allStruct.rdm1_coh_std(:);     % volatility condition (低/高, 根据你自己的字段改)
%confRating  = allStruct.confident(:);   % 如果之后要做 conf，可以用这个

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. Sliding window：算每个 trial 的 mean 和 std
winLen = 10;  % 每个窗口 10 帧
mean_cell = cell(nTrials, 1);
std_cell  = cell(nTrials, 1);

for tr = 1:nTrials
    ev = motion_energy{tr};   % nFrames x 1 double
    if isempty(ev)
        mean_cell{tr} = [];
        std_cell{tr}  = [];
        continue;
    end
    
    nFrames = numel(ev);
    nWin    = nFrames - winLen + 1;
    if nWin <= 0
        mean_cell{tr} = [];
        std_cell{tr}  = [];
        continue;
    end
    
    mVec = nan(1, nWin);
    sVec = nan(1, nWin);
    for w = 1:nWin
        seg = ev(w:w+winLen-1);
        mVec(w) = mean(seg);
        sVec(w) = std(seg);
    end
    
    mean_cell{tr} = mVec;
    std_cell{tr}  = sVec;
end

%% 2. 把 mean/std warp 到固定的 normalized time 轴（比如 40 个 bin）
nBins   = 40;
mean40  = nan(nTrials, nBins);
std40   = nan(nTrials, nBins);

for tr = 1:nTrials
    mVec = mean_cell{tr};
    sVec = std_cell{tr};
    
    if isempty(mVec) || isempty(sVec)
        continue;
    end
    
    x_old = linspace(0, 1, numel(mVec));
    x_new = linspace(0, 1, nBins);
    
    mean40(tr, :) = interp1(x_old, mVec, x_new, 'linear', 'extrap');
    std40(tr, :)  = interp1(x_old, sVec, x_new, 'linear', 'extrap');
end

%% 3. 在每个时间点上，用 STD ~ |MEAN| + coherence 回归，拿 residual STD
%    residual_vol(tr, t) 就是「在这个时间点、这个trial 的 extra noise」

resVol = nan(nTrials, nBins);

for t = 1:nBins
    y = std40(:, t);              % 当前 time bin 的 std
    X_mean = abs(mean40(:, t));   % 当前 bin 的 |mean evidence|
    X_coh  = cohVec;              % coherence（trial-level）
    
    valid = ~isnan(y) & ~isnan(X_mean) & ~isnan(X_coh);
    
    if nnz(valid) < 10
        % 太少 trial 就不回归了
        continue;
    end
    
    X = [ones(nnz(valid), 1), X_mean(valid), X_coh(valid)];  % [截距, |mean|, coh]
    yy = y(valid);
    
    b    = X \ yy;                % 简单 OLS 回归
    yhat = X * b;                 % 可以被 |mean|+coh 解释掉的部分
    resid = yy - yhat;            % extra noise
    
    resVol(valid, t) = resid;
end

%% 4. 只挑 coherence == 0 的 trial
idx_coh0 = (cohVec == 32) & any(~isnan(resVol), 2);

% 这里假设 volVec 里：0=low vol, 1=high vol（如果是 1/2 就改一下）
lowVal  = min(volVec(~isnan(volVec)));  % 比如 0 或 1
highVal = max(volVec(~isnan(volVec)));  % 比如 1 或 2

idx_lowVol_coh0  = idx_coh0 & (volVec == lowVal);
idx_highVol_coh0 = idx_coh0 & (volVec == highVal);

fprintf('Trials with coh==0 & low vol:  %d\n', nnz(idx_lowVol_coh0));
fprintf('Trials with coh==0 & high vol: %d\n', nnz(idx_highVol_coh0));

%% 5. 在两个条件内，算 mean residual volatility 随时间
mean_resVol_low  = nan(1, nBins);
mean_resVol_high = nan(1, nBins);

for t = 1:nBins
    rv = resVol(:, t);
    
    mean_resVol_low(t)  = mean(rv(idx_lowVol_coh0),  'omitnan');
    mean_resVol_high(t) = mean(rv(idx_highVol_coh0), 'omitnan');
end

timeAxis = linspace(0, 1, nBins);   % normalized time 0~1

%% 6. 画图
figure; hold on; box on;

plot(timeAxis, mean_resVol_low,  '-o', 'LineWidth', 1.5);
plot(timeAxis, mean_resVol_high, '-o', 'LineWidth', 1.5);

xlabel('Normalized time (0–1)');
ylabel('Residual volatility (extra noise)');
title('Extra noise over time (coh = 0): low vs high volatility');

legend({sprintf('Low vol (coh = 0)'), sprintf('High vol (coh = 0)')}, ...
       'Location', 'best');

set(gca, 'FontSize', 12);
