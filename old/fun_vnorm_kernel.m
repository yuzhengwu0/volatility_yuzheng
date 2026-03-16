function [resVol_mat, t_norm] = compute_resVol_kernel(motion_energy, coh, nTimeBins, winLen)
% compute_resVol_kernel
% 输入：
%   motion_energy : N x 1 cell，每个 cell 是这一 trial 的 motion energy time series
%   coh           : N x 1 数字，coherence 条件
%   nTimeBins     : 希望 warp 到多少个 normalized time bin（例如 40）
%   winLen        : sliding window 的帧数（例如 10）
%
% 输出：
%   resVol_mat    : N x nTimeBins 的 residual volatility matrix（已 z-score）
%   t_norm        : 1 x nTimeBins 的 normalized time 轴（0~1）

    if nargin < 3 || isempty(nTimeBins)
        nTimeBins = 40;
    end
    if nargin < 4 || isempty(winLen)
        winLen = 10;
    end

    N = numel(motion_energy);
    t_norm = linspace(0, 1, nTimeBins);

    meanME = nan(N, nTimeBins);  % sliding window mean → warp 后
    stdME  = nan(N, nTimeBins);  % sliding window std  → warp 后

    % ---------- 1. 对每个 trial 做：剪掉末尾 0 → sliding window → warp ----------
    for j = 1:N
        me = motion_energy{j};
        if isempty(me)
            continue;
        end

        % === 关键一步：去掉尾部补的 0（或近似 0 的值） ===
        % 这里用 abs(me)>1e-6 找最后一个非零点，你可以根据数据再调阈值
        lastIdx = find(abs(me) > 1e-12, 1, 'last');
        if isempty(lastIdx) || lastIdx <= winLen
            % 全是 0 或者剩的太短就 skip
            continue;
        end
        me = me(1:lastIdx);
        % ================================================

        nFrame = numel(me);
        nWin   = nFrame - winLen + 1;

        if nWin <= 1
            continue;
        end

        m = zeros(nWin,1);
        s = zeros(nWin,1);
        for w = 1:nWin
            seg = me(w : w + winLen - 1);
            m(w) = mean(seg);
            s(w) = std(seg);
        end

        % 这一 trial 自己的时间轴（0~1），再 warp 到统一的 t_norm
        t_orig = linspace(0, 1, nWin);
        meanME(j,:) = interp1(t_orig, m, t_norm, 'linear', 'extrap');
        stdME(j,:)  = interp1(t_orig, s, t_norm, 'linear', 'extrap');
    end

    % ---------- 2. 对每个 time bin 做回归：STD ~ |MEAN| + coh ----------
    resVol_mat = nan(N, nTimeBins);

    for k = 1:nTimeBins
        y_std = stdME(:, k);          % N x 1
        x_mu  = abs(meanME(:, k));    % N x 1

        valid = ~(isnan(y_std) | isnan(x_mu) | isnan(coh));  % 去掉 NaN

        yk = y_std(valid);
        xk = x_mu(valid);
        ck = coh(valid);

        % 如果这一 bin 上几乎没有 variation，就直接跳过
        if numel(yk) < 20 || std(yk) < 1e-6
            continue;
        end

        Xk = [ones(numel(yk),1), xk, ck];

        % 线性回归：STD = a0 + a1*|MEAN| + a2*coh + error
        beta = Xk \ yk;           % 最小二乘
        y_pred = Xk * beta;       % 预测的 STD
        resid  = yk - y_pred;     % 残差 → residual volatility

        tmp = nan(N,1);
        tmp(valid) = resid;
        resVol_mat(:, k) = tmp;
    end

    % ---------- 3. 对 residual volatility 整体做 z-score ----------
    mu = mean(resVol_mat(:), 'omitnan');
    sd = std(resVol_mat(:),  'omitnan');

    resVol_mat = (resVol_mat - mu) ./ sd;
end
