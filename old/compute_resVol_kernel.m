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

    N      = numel(motion_energy);
    t_norm = linspace(0, 1, nTimeBins);

    % 先存 sliding-window mean / std（warp 之后的）
    meanME = nan(N, nTimeBins);  % trial x timeBin
    stdME  = nan(N, nTimeBins);

    % ---------- 1. 对每个 trial：剪掉末尾 0 → sliding window → warp ----------
    for j = 1:N
        me = motion_energy{j};
        if isempty(me)
            continue;
        end

        % 去掉尾部补的 0（或接近 0 的值）
        lastIdx = find(abs(me) > 1e-12, 1, 'last');
        if isempty(lastIdx) || lastIdx <= winLen
            % 全是 0 或者太短，就跳过
            continue;
        end
        me = me(1:lastIdx);

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

        % 这一 trial 自己的时间轴（0~1），再 warp 到统一 t_norm 上
        t_orig     = linspace(0, 1, nWin);
        meanME(j,:) = interp1(t_orig, m, t_norm, 'linear', 'extrap');
        stdME(j,:)  = interp1(t_orig, s, t_norm, 'linear', 'extrap');
    end

    % ---------- 2. 对每个 time bin 做回归：STD ~ |MEAN| + coh ----------
    resVol_mat = nan(N, nTimeBins);

    for k = 1:nTimeBins
        y_std = stdME(:, k);       % N x 1
        x_mu  = abs(meanME(:, k)); % N x 1

        valid = ~(isnan(y_std) | isnan(x_mu) | isnan(coh));

        yk = y_std(valid);
        xk = x_mu(valid);
        ck = coh(valid);

        % 这个时间点几乎没数据/没变化，就跳过
        if numel(yk) < 20 || std(yk) < 1e-6
            continue;
        end

        % 线性回归：STD = a0 + a1*|MEAN| + a2*coh + error
        Xk    = [ones(numel(yk),1), xk, ck];
        beta  = Xk \ yk;          % 最小二乘
        y_pred = Xk * beta;
        resid  = yk - y_pred;     % 残差 = residual volatility

        tmp        = nan(N,1);
        tmp(valid) = resid;
        resVol_mat(:, k) = tmp;
    end

    % ---------- 3. 对 residual volatility 整体做 z-score ----------
    mu = mean(resVol_mat(:), 'omitnan');
    sd = std(resVol_mat(:),  'omitnan');

    resVol_mat = (resVol_mat - mu) ./ sd;
end


%% ==== 时间分辨率的 median-split kernel（简单版）=====================

uniqSubj = unique(subjID);
nSubj    = numel(uniqSubj);
K        = size(resVol_mat, 2);

bias_medKernel = nan(nSubj, K);   % Δbias(t)
sens_medKernel = nan(nSubj, K);   % Δsens(t)

for s = 1:nSubj
    thisSubj = uniqSubj(s);
    idx      = (subjID == thisSubj);

    conf_s = Conf(idx);          % Ns x 1
    cor_s  = Correct(idx);       % Ns x 1
    V_s    = resVol_mat(idx,:);  % Ns x K

    for k = 1:K
        Vk = V_s(:,k);

        % 去掉 NaN
        valid = ~isnan(Vk);
        Vk_k   = Vk(valid);
        conf_k = conf_s(valid);
        cor_k  = cor_s(valid);

        if numel(Vk_k) < 20
            continue;  % 太少就跳过
        end

        % ===== median split: high vol vs low vol =====================
        medV = median(Vk_k);
        isHigh = Vk_k >= medV;
        isLow  = Vk_k <  medV;

        % -------- Bias: overall p(high) 高 vol - 低 vol -------------
        pHigh_highVol = mean(conf_k(isHigh));
        pHigh_lowVol  = mean(conf_k(isLow));

        bias_medKernel(s,k) = pHigh_highVol - pHigh_lowVol;

        % -------- Sensitivity: correct vs error 的 gap 变化 ---------

        % 高 vol：correct / error 的高信心率
        pHigh_highVol_cor  = mean(conf_k(isHigh & cor_k==1));
        pHigh_highVol_err  = mean(conf_k(isHigh & cor_k==0));

        % 低 vol：correct / error 的高信心率
        pHigh_lowVol_cor   = mean(conf_k(isLow  & cor_k==1));
        pHigh_lowVol_err   = mean(conf_k(isLow  & cor_k==0));

        % 如果某一类 trial 太少，会变成 NaN，这没关系
        gap_highVol = pHigh_highVol_cor - pHigh_highVol_err;
        gap_lowVol  = pHigh_lowVol_cor  - pHigh_lowVol_err;

        sens_medKernel(s,k) = gap_highVol - gap_lowVol;
    end
end

%% 画一个被试看看（比如第1个）
exampleIdx = 1;

figure;
subplot(2,1,1);
plot(t_norm, bias_medKernel(exampleIdx,:), '-o');
xlabel('Normalized time');
ylabel('\Delta bias (high - low vol)');
title(sprintf('Subj %d time-resolved bias (median split)', uniqSubj(exampleIdx)));
yline(0,'k--'); grid on;

subplot(2,1,2);
plot(t_norm, sens_medKernel(exampleIdx,:), '-o');
xlabel('Normalized time');
ylabel('\Delta sens (gap_{highVol} - gap_{lowVol})');
title(sprintf('Subj %d time-resolved sensitivity (median split)', uniqSubj(exampleIdx)));
yline(0,'k--'); grid on;
