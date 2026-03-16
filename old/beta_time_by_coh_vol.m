%% beta_time_by_coh_vol.m
% 目标：
% 1) 从 all.motion_energy 做 sliding window，算 mean & std
% 2) 归一化到 0–1 时间轴 (40 bins)
% 3) 在每个时间点上，用 STD ~ |MEAN| + coherence 回归，取 residual STD
% 4) 对以下 6 个条件：
%       coh = 0, 64, 256
%       vol = low (min coh_std), high (max coh_std)
%    在每个时间点上跑 logistic 回归：
%       correct ~ resVol(t) + |meanEv(t)|
%    取 resVol 的回归系数 β_resVol(t)，画成随时间的曲线。

clear; clc;

%% 0. 读入数据
data_path = 'all_with_me.mat';   % 改成你的路径
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;    % nTrials x 1 cell
coh          = allStruct.rdm1_coh(:);      % coherence per trial
coh_std      = allStruct.rdm1_coh_std(:);  % volatility level per trial
correct      = allStruct.correct(:);       % 0/1

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. Sliding window：每个 trial 算 mean evidence & std (volatility)

winLen = 10;   % 每个窗口 10 帧

mean_win = cell(nTrials, 1);   % 每个 trial：窗口 mean evidence
std_win  = cell(nTrials, 1);   % 每个 trial：窗口 std (volatility)

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
    r     = yv - y_hat;        % residual std
    
    resid_STD(valid, b) = r;
end

%% 4. 定义我们要看的 6 个条件：coh × volatility

target_coh = [0 64 256];          % 三个 coherence
vol_levels = unique(coh_std);
vol_low    = min(vol_levels);     % 定义 low volatility
vol_high   = max(vol_levels);     % 定义 high volatility

fprintf('Volatility levels: low = %.4f, high = %.4f\n', vol_low, vol_high);

%% 5. 对每个 (coh, vol) 条件，在每个 time bin 上跑 logistic 回归：
%    correct ~ resVol(t) + |meanEv(t)|，
%    取 resVol 的系数 β_resVol(t)

minTrials = 30;   % 每个条件每个 time 至少 trial 数；太少就跳过

nCoh = numel(target_coh);
beta_all = nan(nCoh, 2, nBins);   % dim: cohIndex x volIndex(1=low,2=high) x time

for ic = 1:nCoh
    c = target_coh(ic);
    
    % low vol 条件
    for iv = 1:2
        if iv == 1
            vVal = vol_low;
            volLabel = 'low';
        else
            vVal = vol_high;
            volLabel = 'high';
        end
        
        idx_cond = (coh == c) & (coh_std == vVal);
        nCond    = sum(idx_cond);
        fprintf('Condition: coh = %3d, vol = %s, nTrials = %d\n', c, volLabel, nCond);
        
        if nCond < minTrials
            warning('Too few trials for coh=%d, vol=%s (n=%d). Results may be unstable.', c, volLabel, nCond);
        end
        
        resVol_cond = resid_STD(idx_cond, :);    % nCond x nBins
        meanEv_cond = MEAN_norm(idx_cond, :);   % nCond x nBins
        correct_cond = correct(idx_cond);       % nCond x 1
        
        for b = 1:nBins
            rv = resVol_cond(:, b);             % residual volatility at time bin b
            mv = meanEv_cond(:, b);            % mean evidence at time bin b
            
            % 去掉 NaN
            valid = ~isnan(rv) & ~isnan(mv) & ~isnan(correct_cond);
            if sum(valid) < 20
                beta_all(ic, iv, b) = NaN;
                continue;
            end
            
            X = [rv(valid), abs(mv(valid))];    % predictors: resVol, |mean|
            y = correct_cond(valid);            % 0/1
            
            % logistic 回归：correct ~ resVol + |mean|
            try
                bHat = glmfit(X, y, 'binomial', 'link', 'logit');
                % bHat(1): intercept, bHat(2): resVol, bHat(3): |mean|
                beta_all(ic, iv, b) = bHat(2);
            catch
                beta_all(ic, iv, b) = NaN;
            end
        end
    end
end

%% 6. 画图：3 (coh) × 2 (vol) 的 subplot，纵轴是 β_resVol(t)

figure;
colors = lines(1);   % 一种颜色就够，这里只画 resVol 的 β

for ic = 1:nCoh
    for iv = 1:2
        subplot(nCoh, 2, (ic-1)*2 + iv); hold on;
        
        beta_curve = squeeze(beta_all(ic, iv, :))';
        if all(isnan(beta_curve))
            title(sprintf('coh = %d, vol = %s (no data)', ...
                  target_coh(ic), ternary(iv==1,'low','high')));
            axis off;
            continue;
        end
        
        % 平滑一下 β 曲线看得更清楚一些（可选）
        beta_smooth = smoothdata(beta_curve, 'gaussian', 3);
        
        plot(t_norm, beta_smooth, 'LineWidth', 2);
        yline(0, '--', 'Color', [0.5 0.5 0.5]);  % 零线
        
        if iv == 1
            vol_str = sprintf('vol = low (%.3f)', vol_low);
        else
            vol_str = sprintf('vol = high (%.3f)', vol_high);
        end
        
        title(sprintf('coh = %d, %s', target_coh(ic), vol_str), 'Interpreter', 'none');
        xlabel('Normalized time');
        ylabel('\beta_{resVol} (effect on correctness)');
        grid on;
        hold off;
    end
end

sgtitle('\beta_{resVol}(t) across time for selected (coh, volatility) conditions');
