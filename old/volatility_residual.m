%% Residual_volatility_kernel_mean_coh.m
% 1) Use all.motion_energy to re-compute sliding-window mean and std for every trial
% 2) Put mean and std onto the same 0–1 normalized time axis
% 3) At each time point, do regression: STD ~ |MEAN| + COHERENCE, and take residual STD
% 4) Use this residual STD to make a Correct vs Incorrect kernel

clear; clc;

%% 0. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;    % 7584 x 1 cell, one cell = one trial
correct_vec   = allStruct.correct(:);       % 1 = correct, 0 = incorrect

% >>> VERY IMPORTANT: change the field name here if needed <<<
% If your coherence field is named differently (e.g., 'coh'), change it.
coh_vec = allStruct.rdm1_coh(:);           % trial-wise coherence for each trial

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. For each trial, do sliding window and compute mean and std (also remove tail zeros)

winLen = 10;                                % window has 10 frames
evidence_strength   = cell(nTrials, 1);     % each trial: sliding-window mean
volatility_strength = cell(nTrials, 1);     % each trial: sliding-window std

tol = 1e-12;                                % very small number, treat as zero

for tr = 1:nTrials
    frames = motion_energy{tr};            % nFrames x 1
    trace  = frames(:)';                   % 1 x nFrames (with padding 0 at the end)
    
    % Find the last non-zero frame, and cut off the padding zeros at the end
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        % If everything is (almost) 0, this trial has no useful data
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    trace_eff = trace(1:last_nz);          % keep only real data frames
    nFrames   = numel(trace_eff);
    
    % If there are fewer frames than one window, skip this trial
    if nFrames < winLen
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    % Number of sliding windows for this trial
    nWin    = nFrames - winLen + 1;
    ev_mean = nan(1, nWin);               % will store mean in each window
    ev_std  = nan(1, nWin);               % will store std in each window
    
    for w = 1:nWin
        % Take one small segment of length winLen
        segment = trace_eff(w : w + winLen - 1);
        % Mean evidence in this window (signed)
        ev_mean(w) = mean(segment);
        % Volatility in this window = std of evidence in this window
        ev_std(w)  = std(segment);
    end
    
    % Save the time series of mean and std for this trial
    evidence_strength{tr}   = ev_mean;
    volatility_strength{tr} = ev_std;
end

%% 2. Put mean & std onto the same 0–1 normalized time axis

nBins  = 40;                               % we want 40 time points from 0 to 1
t_norm = linspace(0, 1, nBins);           % normalized time: 0 = start, 1 = end

MEAN_norm = nan(nTrials, nBins);          % trial x time, normalized mean
STD_norm  = nan(nTrials, nBins);          % trial x time, normalized std

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};        % sliding-window mean for this trial
    sd_tr = volatility_strength{tr};      % sliding-window std for this trial
    
    % If this trial has no data, skip it
    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end
    
    % Make sure mean and std have same length
    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);
    
    % Original time axis for this trial (from 0 to 1, nWin_tr points)
    t_orig = linspace(0, 1, nWin_tr);
    
    % Interpolate mean and std onto the common normalized time axis t_norm
    MEAN_norm(tr, :) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,  :) = interp1(t_orig, sd_tr, t_norm, 'linear');
end

%% 3. At each time point, regress STD ~ |MEAN| + COHERENCE and take residual STD
%    (multiple linear regression, not logistic regression)

resid_STD = nan(size(STD_norm));    % trial x time, residual volatility

for b = 1:nBins
    y  = STD_norm(:, b);            % y = volatility at this time bin (continuous)
    x1 = abs(MEAN_norm(:, b));      % x1 = |mean evidence| at this time bin
    x2 = coh_vec;                   % x2 = coherence for this trial (same for all time bins)
    
    % Use only trials that have real numbers (not NaN)
    mask = ~isnan(y) & ~isnan(x1) & ~isnan(x2);
    if sum(mask) < 10
        % If fewer than 10 trials have data here, skip this time bin
        continue;
    end
    
    % Design matrix with two predictors:
    % X = [constant, |mean|, coherence]
    X = [ones(sum(mask), 1), x1(mask), x2(mask)];
    
    % Multiple linear regression: y = beta0 + beta1 * |mean| + beta2 * coherence + error
    beta = X \ y(mask);
    
    % Predicted std from mean and coherence
    % y_hat = beta0 + beta1 * |mean| + beta2 * coherence
    y_hat = X * beta;
    
    % residual = real std - predicted std
    tmp       = nan(size(y));
    tmp(mask) = y(mask) - y_hat;
    
    % Save residual volatility at this time bin for all trials
    resid_STD(:, b) = tmp;
end

%% 4. Use residual volatility to make Correct vs Incorrect kernel

idxC = (correct_vec == 1);         % trials where response is correct
idxI = (correct_vec == 0);         % trials where response is incorrect

% Average residual volatility over Correct trials, at each time bin
mean_resid_C = mean(resid_STD(idxC, :), 1, 'omitnan');
% Average residual volatility over Incorrect trials, at each time bin
mean_resid_I = mean(resid_STD(idxI, :), 1, 'omitnan');

% Kernel = Correct minus Incorrect, at each time bin
kernel_resid = mean_resid_C - mean_resid_I;

% ---- Plot results ----
figure;

% Top plot: residual volatility over time for Correct vs Incorrect
subplot(2,1,1);
plot(t_norm, mean_resid_C, '-g', 'LineWidth', 1.2); hold on;
plot(t_norm, mean_resid_I, '-m', 'LineWidth', 1.2);
xlabel('Normalized time within trial (0 = start, 1 = end)');
ylabel('Residual volatility');
legend({'Correct','Incorrect'}, 'Location','best');
title('Residual volatility over time (after regressing out |mean evidence| and coherence)');
grid on;

% Bottom plot: kernel = Correct - Incorrect
subplot(2,1,2);
plot(t_norm, kernel_resid, '-k', 'LineWidth', 1.2); hold on;
yline(0, '--');                             % zero line for reference
xlabel('Normalized time within trial (0 = start, 1 = end)');
ylabel('Residual volatility (Correct - Incorrect)');
title('Residual VOLATILITY accuracy kernel (controlling for mean & coherence)');
grid on;



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
