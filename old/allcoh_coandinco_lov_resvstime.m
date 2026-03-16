%% plot_resVol_lowVol_corr_vs_incorr.m
% 同时画：
%   - low volatility (coh_std == 0) + correct
%   - low volatility (coh_std == 0) + incorrect
% 在 6 个 coherence 水平 (0,32,64,128,256,512) 下，
% residual volatility 随 normalized time (0–1) 的时间曲线，
% 并让两张图共用同一组 y 轴范围，方便比较。

clear; clc;

%% 0. 读入数据
data_path = 'all_with_me.mat';   % 改成你的路径
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell
coh          = allStruct.rdm1_coh(:);      % coherence
coh_std      = allStruct.rdm1_coh_std(:);  % coherence std = volatility level
correct      = allStruct.correct(:);       % 0/1

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. Sliding window：每个 trial 算 mean evidence & std
winLen = 10;   % 每个窗口 10 帧

mean_win = cell(nTrials, 1);
std_win  = cell(nTrials, 1);

for tr = 1:nTrials
    frames = motion_energy{tr};   % nFrames x 1
    if isempty(frames)
        mean_win{tr} = [];
        std_win{tr}  = [];
        continue;
    end
    
    trace = frames(:)';           % 1 x nFrames
    
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
    
    X = [ones(sum(valid), 1), abs(mv), cv];   % [1, |mean|, coh]
    beta  = X \ yv;
    y_hat = X * beta;
    r     = yv - y_hat;
    resid_STD(valid, b) = r;
end

%% 4. 为每个 coherence，分别算 correct / incorrect 的曲线
coh_levels = [0 32 64 128 256 512];
nLevels    = numel(coh_levels);
minTrials  = 5;   % 少于这个就不画

Y_corr = nan(nLevels, nBins);   % 每行一条 curve
Y_inc  = nan(nLevels, nBins);
nCorr  = zeros(nLevels, 1);
nInc   = zeros(nLevels, 1);

for i = 1:nLevels
    c = coh_levels(i);
    idx_coh     = (coh == c);
    idx_lowVol  = (coh_std == 0);     % low volatility 条件
    idx_corr    = (correct == 1);
    idx_inc     = (correct == 0);
    
    % correct
    idx_use_corr = idx_coh & idx_lowVol & idx_corr;
    nCorr(i)     = sum(idx_use_corr);
    fprintf('Coh=%3d, lowVol, correct trials = %d\n', c, nCorr(i));
    if nCorr(i) >= minTrials
        resid_sub = resid_STD(idx_use_corr, :);
        m = mean(resid_sub, 1, 'omitnan');
        Y_corr(i, :) = smoothdata(m, 'gaussian', 5);
    end
    
    % incorrect
    idx_use_inc = idx_coh & idx_lowVol & idx_inc;
    nInc(i)     = sum(idx_use_inc);
    fprintf('Coh=%3d, lowVol, incorrect trials = %d\n', c, nInc(i));
    if nInc(i) >= minTrials
        resid_sub = resid_STD(idx_use_inc, :);
        m = mean(resid_sub, 1, 'omitnan');
        Y_inc(i, :) = smoothdata(m, 'gaussian', 5);
    end
end

%% 5. 统一 y 轴范围
allY = [Y_corr(:); Y_inc(:)];
allY = allY(~isnan(allY));
ymin = min(allY);
ymax = max(allY);
pad  = 0.05 * (ymax - ymin);   % 边缘多留一点
yl   = [ymin - pad, ymax + pad];

%% 6. 画图：左 correct，右 incorrect，共用同一 ylim
colors = lines(nLevels);

figure;

% ---- correct ----
subplot(1, 2, 1); hold on;
for i = 1:nLevels
    if isnan(Y_corr(i, 1)); continue; end
    plot(t_norm, Y_corr(i, :), 'LineWidth', 2, 'Color', colors(i, :));
end
xlabel('Normalized time (0–1)');
ylabel('Residual volatility');
title('Low-volatility & CORRECT trials');
legend(string(coh_levels), 'Location', 'best');
ylim(yl);
grid on;

% ---- incorrect ----
subplot(1, 2, 2); hold on;
for i = 1:nLevels
    if isnan(Y_inc(i, 1)); continue; end
    plot(t_norm, Y_inc(i, :), 'LineWidth', 2, 'Color', colors(i, :));
end
xlabel('Normalized time (0–1)');
ylabel('Residual volatility');
title('Low-volatility & INCORRECT trials');
legend(string(coh_levels(coh_levels ~= 512)), 'Location', 'best'); % 如果 512 太少会被 NaN
ylim(yl);
grid on;
