%% resVol_time_conf_kernels.m
% 1) 从 all.motion_energy 重新算 sliding-window mean / std
% 2) warp 到 0–1 的 normalized time 轴
% 3) 在每个 time bin 上做回归：STD ~ |MEAN|，只去掉 evidence 强度（不再去掉 coherence）
% 4) 得到 residual volatility 矩阵 resVol_mat(trial x timeBin)
% 5) 先画一个 Correct vs Incorrect 的 residual volatility kernel（group level）
% 6) 再做按时间的 median-split bias & sensitivity kernel（per subject）

clear; clc;

%% 0. Load data -----------------------------------------------------------
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;      % N x 1 cell
Correct       = allStruct.correct(:);         % N x 1, 1 = correct, 0 = error
coh_vec       = allStruct.rdm1_coh(:);        % 如果后面不需要可以不用
subjID        = allStruct.group(:);          % subject ID

% confidence → 高/低
Conf_raw  = allStruct.confidence(:);         % 0–1 连续
Conf_raw  = double(Conf_raw);
th        = 0.5;                             % 阈值 >=0.5 = 高信心
Conf      = double(Conf_raw >= th);          % 高=1, 低=0

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. Sliding-window mean & std（去掉尾部 0）--------------------------
winLen = 10;                                  % window 长度（帧）
tol    = 1e-12;                               % 判定“非 0”的阈值

evidence_strength   = cell(nTrials, 1);       % 每个 trial：sliding mean
volatility_strength = cell(nTrials, 1);       % 每个 trial：sliding std

for tr = 1:nTrials
    frames = motion_energy{tr};               % nFrames x 1
    trace  = frames(:)';                      % 1 x nFrames

    % 去掉尾部补的 0
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);

    if nFrames < winLen
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    nWin  = nFrames - winLen + 1;
    m_win = nan(1, nWin);
    s_win = nan(1, nWin);

    for w = 1:nWin
        seg      = trace_eff(w : w + winLen - 1);
        m_win(w) = mean(seg);
        s_win(w) = std(seg);
    end

    evidence_strength{tr}   = m_win;          % signed mean
    volatility_strength{tr} = s_win;          % std in window
end

%% 2. Warp 到统一的 normalized time 轴 ---------------------------------
nBins  = 40;                                  % = nTimeBins
t_norm = linspace(0, 1, nBins);               % 0 ~ 1

MEAN_norm = nan(nTrials, nBins);              % trial x time
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};

    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end

    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    t_orig = linspace(0, 1, nWin_tr);

    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

%% 3. 每个 time bin 做回归：STD ~ |MEAN|（不再包含 coherence）----------
%    得到 residual volatility matrix: resVol_mat(trial x timeBin)

resVol_mat = nan(size(STD_norm));   % N x nBins

for b = 1:nBins
    y  = STD_norm(:, b);            % volatility at time bin b
    x1 = abs(MEAN_norm(:, b));      % |mean evidence|

    mask = ~isnan(y) & ~isnan(x1);
    if sum(mask) < 10
        continue;
    end

    X    = [ones(sum(mask),1), x1(mask)];   % [截距, |mean|]
    y_use = y(mask);

    beta  = X \ y_use;                      % OLS 回归
    y_hat = X * beta;
    resid = y_use - y_hat;

    tmp       = nan(size(y));
    tmp(mask) = resid;
    resVol_mat(:, b) = tmp;
end

% （可选）整体 z-score，一般对 median split 影响不大，这里可以保留 / 去掉
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_mat = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d time bins.\n', ...
        size(resVol_mat,1), size(resVol_mat,2));


% 加这一行：给“残差波动”一个统一名字
resid_STD = resVol_mat;

%% 4. Group-level Correct vs Incorrect residual volatility kernel -------
idxC = (Correct == 1);
idxI = (Correct == 0);

mean_resid_C = mean(resVol_mat(idxC, :), 1, 'omitnan');
mean_resid_I = mean(resVol_mat(idxI, :), 1, 'omitnan');

kernel_resid = mean_resid_C - mean_resid_I;

figure;
subplot(2,1,1);
plot(t_norm, mean_resid_C, '-g', 'LineWidth', 1.2); hold on;
plot(t_norm, mean_resid_I, '-m', 'LineWidth', 1.2);
xlabel('Normalized time within trial');
ylabel('Residual volatility');
legend({'Correct','Incorrect'}, 'Location','best');
title('Residual volatility over time (Correct vs Incorrect)');
grid on;

subplot(2,1,2);
plot(t_norm, kernel_resid, '-k', 'LineWidth', 1.2); hold on;
yline(0,'--');
xlabel('Normalized time within trial');
ylabel('Correct - Incorrect');
title('Residual VOLATILITY accuracy kernel');
grid on;

%% ==== 时间分辨率的 median-split kernel（按 coh 条件化）=====================

% 我们把 resid_STD 当成 time-resolved volatility：
% rows = trials, cols = time bins
% 用 residual volatility 作为 time-resolved volatility：
resVol_time = resid_STD;        % or: resVol_time = resVol_mat;
K          = size(resVol_time,2);


uniqSubj = unique(subjID);
nSubj    = numel(uniqSubj);
uniqCoh  = unique(coh_vec);     % 所有 coherence 水平

bias_medKernel_coh = nan(nSubj, K);   % Δbias(t): highVol - lowVol
sens_medKernel_coh = nan(nSubj, K);   % Δsens(t): (gap_highVol - gap_lowVol)

for s = 1:nSubj
    thisSubj = uniqSubj(s);
    idx_s    = (subjID == thisSubj);
    
    conf_s = Conf(idx_s);          % Ns x 1, 0/1 high vs low confidence
    cor_s  = Correct(idx_s);       % Ns x 1, 0/1 correct vs error
    V_s    = resVol_time(idx_s,:);  % Ns x K, time-resolved residual vol
    coh_s  = coh_vec(idx_s);       % Ns x 1, coherence for each trial
    
    % 对每一个时间点单独做
    for k = 1:K
        Vk_all = V_s(:, k);        % Ns x 1, 这个 time bin 的 volatility
        
        % 我们会在每个 coh 里单独做 median split，然后把结果加权平均
        num_bias   = 0;   % 累加 Δbias 的 “分子”
        den_bias   = 0;   % 累加 Δbias 的 “权重”（trial 数）
        num_sens   = 0;   % 累加 Δsens 的 “分子”
        den_sens   = 0;   % 累加 Δsens 的 “权重”（trial 数）
        
        for cidx = 1:numel(uniqCoh)
            thisCoh = uniqCoh(cidx);
            
            % 当前被试 + 当前 coh + 有效 Vk 的 trial
            mask = (~isnan(Vk_all)) & (coh_s == thisCoh);
            if sum(mask) < 20
                continue;   % 这个 coh 在该时间点试次太少，就跳过
            end
            
            Vk     = Vk_all(mask);
            conf_k = conf_s(mask);
            cor_k  = cor_s(mask);
            
            % ----- 在这个 coh 里做一次 median split: high/low vol -----
            medV   = median(Vk);
            isHigh = Vk >= medV;
            isLow  = Vk <  medV;
            
            % 如果其中一组太少，也跳过
            if sum(isHigh) < 5 || sum(isLow) < 5
                continue;
            end
            
            % ============ Bias part ============ %
            pHigh_highVol = mean(conf_k(isHigh));   % p(high | high vol, this coh)
            pHigh_lowVol  = mean(conf_k(isLow));    % p(high | low vol,  this coh)
            
            delta_bias_c  = pHigh_highVol - pHigh_lowVol;  % 这一 coh 的 Δbias
            
            % 用这一 coh 的 trial 数作为权重
            w_bias = sum(mask);
            num_bias = num_bias + w_bias * delta_bias_c;
            den_bias = den_bias + w_bias;
            
            % ============ Sensitivity part ============ %
            % 高 vol：correct / error 的高信心率
            pHigh_highVol_cor = mean(conf_k(isHigh & cor_k == 1));
            pHigh_highVol_err = mean(conf_k(isHigh & cor_k == 0));
            
            % 低 vol：correct / error 的高信心率
            pHigh_lowVol_cor  = mean(conf_k(isLow  & cor_k == 1));
            pHigh_lowVol_err  = mean(conf_k(isLow  & cor_k == 0));
            
            % 如果某些 cell 没 trial，会变成 NaN，这种 coh 我们就不计入 sens 的平均
            if any(isnan([pHigh_highVol_cor, pHigh_highVol_err, ...
                          pHigh_lowVol_cor,  pHigh_lowVol_err]))
                continue;
            end
            
            gap_highVol = pHigh_highVol_cor - pHigh_highVol_err;
            gap_lowVol  = pHigh_lowVol_cor  - pHigh_lowVol_err;
            
            delta_sens_c = gap_highVol - gap_lowVol;   % 这一 coh 的 Δsens
            
            % sensitivity 这边可以用 “有完整四个 cell 的 trial 数” 做权重
            w_sens = sum( (isHigh & (cor_k==1)) | ...
                          (isHigh & (cor_k==0)) | ...
                          (isLow  & (cor_k==1)) | ...
                          (isLow  & (cor_k==0)) );
            num_sens = num_sens + w_sens * delta_sens_c;
            den_sens = den_sens + w_sens;
        end
        
        % 把各个 coh 的结果加权平均
        if den_bias > 0
            bias_medKernel_coh(s,k) = num_bias / den_bias;
        end
        if den_sens > 0
            sens_medKernel_coh(s,k) = num_sens / den_sens;
        end
    end
end

%% 画 6 张图：上 3 = Δbias，下 3 = Δsens（coh-stratified），每一行 y 轴对齐
figure;

% 1) 先算这一套 kernel 的全局 y 轴范围（分别算 bias 和 sens）
yBias_all = bias_medKernel_coh(:);
ySens_all = sens_medKernel_coh(:);

yBias_min = min(yBias_all, [], 'omitnan');
yBias_max = max(yBias_all, [], 'omitnan');

ySens_min = min(ySens_all, [], 'omitnan');
ySens_max = max(ySens_all, [], 'omitnan');

% 2) 逐个被试画图
for s = 1:nSubj
    % ---------- 上面一行：Δbias ----------
    subplot(2, nSubj, s);   % 第 1 行，第 s 列
    plot(t_norm, bias_medKernel_coh(s,:), '-o'); 
    hold on;
    yline(0, 'k--');
    ylim([yBias_min, yBias_max]);   % 这一行所有 subplot 统一 y 轴

    title(sprintf('Subj %d  \\Delta bias', uniqSubj(s)));
    xlabel('Normalized time');
    ylabel('\Delta bias (high - low vol)');
    grid on;

    % ---------- 下面一行：Δsens ----------
    subplot(2, nSubj, nSubj + s);   % 第 2 行，第 s 列
    plot(t_norm, sens_medKernel_coh(s,:), '-o'); 
    hold on;
    yline(0, 'k--');
    ylim([ySens_min, ySens_max]);   % 这一行所有 subplot 统一 y 轴

    title(sprintf('Subj %d  \\Delta sens', uniqSubj(s)));
    xlabel('Normalized time');
    ylabel('\Delta sens');
    grid on;
end


% ===== coh-stratified 版本：三个被试，各 1 张 bias+sens 图 =====

figure;

% 先算这套 kernel 的全局 y 轴范围（单独算，不跟 simple 版混）
y_all_coh = [bias_medKernel_coh(:); sens_medKernel_coh(:)];
y_min_coh = min(y_all_coh, [], 'omitnan');
y_max_coh = max(y_all_coh, [], 'omitnan');

for s = 1:nSubj   % 你现在是 3 个被试，这里就是 3 次
    subplot(1, nSubj, s);      % 一行三个 subplot
    plot(t_norm, bias_medKernel_coh(s,:), '-o'); hold on;
    plot(t_norm, sens_medKernel_coh(s,:), '-o');
    yline(0,'k--');
    ylim([y_min_coh, y_max_coh]);   % coh-residual 这套用统一纵轴

    title(sprintf('Subj %d (coh-stratified)', uniqSubj(s)));
    xlabel('Normalized time');
    ylabel('\Delta bias / \Delta sens');
    legend({'\Delta bias_{coh}','\Delta sens_{coh}'}, 'Location','best');
    grid on;
end


%% 5. 时间分辨率的 median-split bias & sensitivity kernel（per subject）
uniqSubj = unique(subjID);
nSubj    = numel(uniqSubj);
K        = size(resVol_mat, 2);

bias_medKernel = nan(nSubj, K);   % Δbias(t) = pHigh(highVol) - pHigh(lowVol)
sens_medKernel = nan(nSubj, K);   % Δsens(t) = [gap_highVol - gap_lowVol]

for s = 1:nSubj
    thisSubj = uniqSubj(s);
    idx      = (subjID == thisSubj);

    conf_s = Conf(idx);           % Ns x 1, 0/1
    cor_s  = Correct(idx);        % Ns x 1
    V_s    = resVol_mat(idx, :);  % Ns x K

    for k = 1:K
        Vk = V_s(:,k);

        % 去掉 NaN
        valid = ~isnan(Vk);
        Vk_k   = Vk(valid);
        conf_k = conf_s(valid);
        cor_k  = cor_s(valid);

        if numel(Vk_k) < 20
            continue;             % 数据太少就跳过
        end

        % ===== median split: high vol vs low vol =====================
        medV  = median(Vk_k);
        isHigh = Vk_k >= medV;
        isLow  = Vk_k <  medV;

        % -------- Bias: overall p(high) 高 vol - 低 vol -------------
        pHigh_highVol = mean(conf_k(isHigh));
        pHigh_lowVol  = mean(conf_k(isLow));

        bias_medKernel(s,k) = pHigh_highVol - pHigh_lowVol;

        % -------- Sensitivity: correct vs error 的 gap 变化 ---------
        pHigh_highVol_cor = mean(conf_k(isHigh & cor_k==1));
        pHigh_highVol_err = mean(conf_k(isHigh & cor_k==0));

        pHigh_lowVol_cor  = mean(conf_k(isLow  & cor_k==1));
        pHigh_lowVol_err  = mean(conf_k(isLow  & cor_k==0));

        gap_highVol = pHigh_highVol_cor - pHigh_highVol_err;
        gap_lowVol  = pHigh_lowVol_cor  - pHigh_lowVol_err;

        sens_medKernel(s,k) = gap_highVol - gap_lowVol;
    end
end

% %% 6. 画一个被试的 time-resolved bias / sensitivity kernel -------------
% % ===== 多被试 + 统一 y 轴 =====
% figure;
% 
% % 先把所有被试的 bias / sens 拼在一起，算全局的 y 轴范围
% y_all = [bias_medKernel(:); sens_medKernel(:)];
% y_min = min(y_all, [], 'omitnan');
% y_max = max(y_all, [], 'omitnan');
% 
% for s = 1:nSubj
%     subplot(1, nSubj, s);  % 如果以后被试变多，可以改成 2xN 等
%     plot(t_norm, bias_medKernel(s,:), '-o'); hold on;
%     plot(t_norm, sens_medKernel(s,:), '-o');
%     yline(0,'k--');
%     ylim([y_min y_max]);   % 统一纵轴（y-axis）
% 
%     title(sprintf('Subj %d', uniqSubj(s)));
%     xlabel('Normalized time');
%     ylabel('\Delta bias / \Delta sens');
%     legend({'\Delta bias','\Delta sens'}, 'Location','best');
%     grid on;
% end


%% === Logistic regression confidence kernel (group-level) ===
% 目标：
% 对每个 time bin k，用 trial-by-trial residual volatility 预测 Conf，
% 并控制 Correct + coherence，得到 Δp(high conf) 随时间的 kernel。

[N, K] = size(resVol_time);

beta_vol   = nan(K,1);  % 每个 time bin 上 volatility 的 logistic 系数
kernel_dp  = nan(K,1);  % 每个 time bin 上 Δp(high conf)（+1SD vs -1SD vol）

for k = 1:K
    Vk = resVol_time(:, k);   % N x 1，这个 time bin 的 residual volatility
    
    % 有效 trial：V、Conf、Correct、coh 都不是 NaN
    mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(coh_vec);
    if sum(mask) < 30
        continue;             % trial 太少就跳过
    end
    
    y   = Conf(mask);         % 0/1
    C   = Correct(mask);      % 0/1
    coh = coh_vec(mask);      % coherence
    V   = Vk(mask);           % residual volatility
    
    % 建议把连续变量 z-score，一会儿方便解释成 “1 SD” 的变化
    V_z   = (V   - mean(V))   ./ std(V);
    coh_z = (coh - mean(coh)) ./ std(coh);
    
    % 设计矩阵（不含截距，glmfit 会自己加 intercept）
    X = [C, coh_z, V_z];   % 列顺序：Correct, coh_z, V_z
    
    % logistic regression: Conf ~ Correct + coh + V_resid
    % b(1) = intercept, b(2) = Correct, b(3) = coh_z, b(4) = V_z
    b = glmfit(X, y, 'binomial', 'link', 'logit');
    
    % volatility 的系数
    beta_vol(k) = b(4);
    
    % ==== 把系数转成 Δp(high conf) （+1SD vs -1SD volatility）====
    % 选一个“代表性”的 Correct / coh 水平来算预测概率
    C0   = mean(C);       % 平均 correctness（也可以设成 0.5）
    coh0 = 0;             % 因为 coh_z 已经 z-score 过，所以 mean≈0
    
    V_hi = +1;            % +1 SD volatility
    V_lo = -1;            % -1 SD volatility
    
    eta_hi = b(1) + b(2)*C0 + b(3)*coh0 + b(4)*V_hi;
    eta_lo = b(1) + b(2)*C0 + b(3)*coh0 + b(4)*V_lo;
    
    p_hi  = 1 ./ (1 + exp(-eta_hi));
    p_lo  = 1 ./ (1 + exp(-eta_lo));
    
    kernel_dp(k) = p_hi - p_lo;   % 这个 time bin 上 volatility ↑1SD 带来的 p(high) 提升
end

%% 画图：时间 × Δp(high conf)
figure;

subplot(2,1,1);
plot(t_norm, kernel_dp, '-o', 'LineWidth', 1.2); hold on;
yline(0, 'k--');
xlabel('Normalized time within trial');
ylabel('\Delta p(high conf)');
title('Volatility \rightarrow confidence kernel (frame level)');
grid on;

subplot(2,1,2);
plot(t_norm, beta_vol, '-o', 'LineWidth', 1.2); hold on;
yline(0, 'k--');
xlabel('Normalized time within trial');
ylabel('\beta_{vol}(t)');
title('Logistic coefficient of residual volatility');
grid on;
