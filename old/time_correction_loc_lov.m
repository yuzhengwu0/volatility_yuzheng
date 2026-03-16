%% plot_resVol_6coh_lowVol_correct.m
% 目标：
% 在 6 个 coherence 水平 (0,32,64,128,256,512) 下，
% 只取 low volatility 条件 + 答对(correct=1) 的 trial，
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

%% 1. Sliding window: 每个 trial 计算窗口 mean & std

winLen = 10;   % 每个窗口 10 帧

mean_win = cell(nTrials, 1);  % 每个 trial 的窗口 mean evidence
std_win  = cell(nTrials, 1);  % 每个 trial 的窗口 std (volatility)

for tr = 1:nTrials
    frames = motion_energy{tr};       % nFrames x 1
    if isempty(frames)
        mean_win{tr} = [];
        std_win{tr}  = [];
        continue;
    end
    
    trace = frames(:)';               % 1 x nFrames
    
    % 去掉结尾 padding 0
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

%% 2. 把 mean & std 插值到统一的 normalized time 轴 (40 bins)

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

%% 3. 在每个时间点回归：STD ~ |MEAN| + coherence，拿 residual STD

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
    r     = yv - y_hat;        % residual volatility
    
    resid_STD(valid, b) = r;
end

%% 4. 为 6 个 coherence 水平画曲线（low volatility + correct）

coh_levels = [0 32 64 128 256 512];
nLevels    = numel(coh_levels);

figure; hold on;

colors = lines(nLevels);   % 6 种颜色

for i = 1:nLevels
    c = coh_levels(i);
    
    % 找这个 coherence 下有哪些 volatility level
    idx_coh = (coh == c);
    vols    = unique(coh_std(idx_coh));
    
    if isempty(vols)
        warning('No trials found for coherence = %d', c);
        continue;
    end
    
    % 取该 coherence 下最小的 std 作为 low volatility（如果你的设计里是 std=0 就会选到 0）
    vol_low = min(vols);
    
    idx_use = idx_coh & (coh_std == vol_low) & (correct == 1);
    nUse    = sum(idx_use);
    fprintf('Coh=%3d, lowVol=%.4f, correct trials = %d\n', c, vol_low, nUse);
    
    if nUse < 5
        warning('Too few trials for coherence=%d, skip plotting.', c);
        continue;
    end
    
    resid_sub = resid_STD(idx_use, :);            % nUse x nBins
    mean_resid = mean(resid_sub, 1, 'omitnan');   % 1 x nBins
    mean_resid_smooth = smoothdata(mean_resid, 'gaussian', 5);
    
    plot(t_norm, mean_resid_smooth, 'LineWidth', 2, ...
         'Color', colors(i, :));
end

xlabel('Normalized time (0–1)');
ylabel('Residual volatility');
title('Low-volatility & correct trials at different coherence levels');
legend(string(coh_levels), 'Location', 'best');
grid on; hold off;
