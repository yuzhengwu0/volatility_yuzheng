%% plot_resVol_6coh_lowVol_incorrect.m
% 目标：
% 在 6 个 coherence 水平 (0,32,64,128,256,512) 下，
% 只取 low volatility 条件 (这里用 coh_std == 0) + incorrect (=0) 的 trial，
% 画出 residual volatility 随 normalized time (0–1) 的曲线（共 6 条）。

clear; clc;

%% 0. 读入数据
data_path = 'all_with_me.mat';   % 改成你的 mat 路径
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell
coh          = allStruct.rdm1_coh(:);      % coherence per trial
coh_std      = allStruct.rdm1_coh_std(:);  % coherence std = volatility level
correct      = allStruct.correct(:);       % 0/1, 是否答对

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. Sliding window：每个 trial 算 mean evidence & std

winLen = 10;   % 每个窗口 10 帧

mean_win = cell(nTrials, 1);
std_win  = cell(nTrials, 1);

for tr = 1:nTrials
    frames = motion_energy{tr};     % nFrames x 1
    if isempty(frames)
        mean_win{tr} = [];
        std_win{tr}  = [];
        continue;
    end
    
    trace = frames(:)';             % 1 x nFrames
    
    % 去掉结尾 padding 的 0
    tol     = 1e-12;
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

%% 2. 插值到统一的 normalized time 轴 (40 bins)

nBins  = 40;
t_norm = linspace(0, 1, nBins);

MEAN_norm = nan(nTrials, nBins);
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mw = mean_win{tr};
    sw = std_win{tr};
    
    if isempty(mw) || numel(mw) < 2
        continue;
    end
    
    nW     = numel(mw);
    t_orig = linspace(0, 1, nW);
    
    MEAN_norm(tr, :) = interp1(t_orig, mw, t_norm, 'linear', 'extrap');
    STD_norm(tr, :)  = interp1(t_orig, sw, t_norm, 'linear', 'extrap');
end

%% 3. 每个时间点回归：STD ~ |MEAN| + coherence，拿 residual STD

resid_STD = nan(size(STD_norm));   % nTrials x nBins

for b = 1:nBins
    y = STD_norm(:, b);
    
    valid = ~isnan(y);
    if sum(valid) < 10
        continue;
    end
    
    yv = y(valid);
    mv = MEAN_norm(valid, b);
    cv = coh(valid);
    
    % 设计矩阵：常数 + |mean| + coherence
    X = [ones(sum(valid), 1), abs(mv), cv];
    
    beta  = X \ yv;
    y_hat = X * beta;
    r     = yv - y_hat;
    
    resid_STD(valid, b) = r;
end

%% 4. 为 6 个 coherence 水平画 "incorrect + low volatility" 曲线

coh_levels = [0 32 64 128 256 512];
nLevels    = numel(coh_levels);

figure; hold on;
colors = lines(nLevels);

for i = 1:nLevels
    c = coh_levels(i);
    
    % 该 coherence 下的 trial
    idx_coh = (coh == c);
    
    % low volatility：这里直接用 coh_std == 0
    idx_lowVol = (coh_std == 0);
    
    % incorrect trials
    idx_incorrect = (correct == 0);
    
    % 综合条件
    idx_use = idx_coh & idx_lowVol & idx_incorrect;
    nUse    = sum(idx_use);
    fprintf('Coh=%3d, lowVol(coh_std=0), incorrect trials = %d\n', c, nUse);
    
    if nUse < 5
        warning('Too few trials for coherence=%d, skip plotting.', c);
        continue;
    end
    
    resid_sub = resid_STD(idx_use, :);             % nUse x nBins
    mean_resid = mean(resid_sub, 1, 'omitnan');    % 1 x nBins
    mean_resid_smooth = smoothdata(mean_resid, 'gaussian', 5);
    
    plot(t_norm, mean_resid_smooth, 'LineWidth', 2, ...
         'Color', colors(i, :));
end

xlabel('Normalized time (0–1)');
ylabel('Residual volatility');
title('Low volatility (coh\_std=0) & Incorrect trials at different coherence levels');
legend(string(coh_levels), 'Location', 'best');
grid on; hold off;
