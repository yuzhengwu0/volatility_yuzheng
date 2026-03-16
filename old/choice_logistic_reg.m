%% choice_logit_mean_resVol.m
% Goal: Use early/mid/late mean evidence + residual volatility
%       to predict p(response = right)
%
% Steps:
% 1) From all.motion_energy do sliding window, compute mean & std
% 2) Normalize mean / std to a 0–1 time axis (40 bins)
% 3) At each time bin, run STD ~ |MEAN| + coherence, take residual STD (extra volatility)
% 4) Split time into early / mid / late, compute mean evidence and residual volatility in each period
% 5) Logistic regression: resp_right (1=right, 0=left) ~ mean(E/M/L) + resVol(E/M/L)

clear; clc;

%% 0. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell (right positive, left negative)
resp_vec      = allStruct.req_resp(:);    % 1 = right, 2 = left

%%% <<< NEW: coherence vector (one value per trial)
coh_vec       = allStruct.rdm1_coh(:);    % 7584 x 1 double, physical coherence

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. For each trial, do sliding window and compute mean & std (after removing tail zeros)

winLen = 10;          % window length: 10 frames
tol    = 1e-12;       % threshold to decide "is this 0" (avoid tiny numbers)

evidence_strength   = cell(nTrials, 1);   % each trial: window mean
volatility_strength = cell(nTrials, 1);   % each trial: window std

for tr = 1:nTrials
    frames = motion_energy{tr};    % nFrames x 1
    trace  = frames(:)';           % 1 x nFrames (may contain padding zeros)
    
    % Remove padding zeros at the end: find the last non-zero
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);
    
    % If this trial is shorter than the window, skip it
    if nFrames < winLen
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    nWin    = nFrames - winLen + 1;
    ev_mean = nan(1, nWin);
    ev_std  = nan(1, nWin);
    
    for w = 1:nWin
        segment    = trace_eff(w : w + winLen - 1);
        ev_mean(w) = mean(segment);   % mean signed evidence in this window
        ev_std(w)  = std(segment);    % volatility in this window
    end
    
    evidence_strength{tr}   = ev_mean;
    volatility_strength{tr} = ev_std;
end

%% 2. Normalize each trial to a common 0–1 time axis

nBins  = 40;                    % number of time bins we want
t_norm = linspace(0, 1, nBins); % 0 = trial start, 1 = trial end

MEAN_norm = nan(nTrials, nBins);   % trial x time
STD_norm  = nan(nTrials, nBins);   % trial x time

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};
    
    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end
    
    % Just in case lengths differ, take the shorter one
    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);
    
    t_orig = linspace(0, 1, nWin_tr);   % this trial’s own 0–1 progress
    
    MEAN_norm(tr, :) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,  :) = interp1(t_orig, sd_tr, t_norm, 'linear');
end

%% 3. At each time bin, run STD ~ |MEAN| + coherence → get residual volatility

resid_STD = nan(size(STD_norm));   % trial x time

for b = 1:nBins
    y = STD_norm(:, b);           % volatility
    m = abs(MEAN_norm(:, b));     % absolute value of evidence strength at this time bin
    c = coh_vec;                  % physical coherence of this trial (same for all bins)
    
    % mask: remove NaNs
    mask = ~isnan(y) & ~isnan(m) & ~isnan(c);
    if sum(mask) < 10
        continue;                 % if too few valid trials, skip this bin
    end
    
    % Design matrix: constant + |mean| + coherence
    X    = [ones(sum(mask),1), m(mask), c(mask)];
    beta = X \ y(mask);           % least squares: y = b0 + b1*|mean| + b2*coh
    
    y_hat = X * beta;             % expected std given mean & coh
    
    tmp       = nan(size(y));
    tmp(mask) = y(mask) - y_hat;  % residual = actual std - expected std
    resid_STD(:, b) = tmp;
end

%% 4. Split time into Early / Mid / Late, take mean evidence + residual volatility

% Time split (40 bins → 3 segments)
early_idx = 1:13;    % about 0 ~ 0.33
mid_idx   = 14:26;   % about 0.33 ~ 0.66
late_idx  = 27:40;   % about 0.66 ~ 1

mean_E_mean = mean(MEAN_norm(:, early_idx), 2, 'omitnan');
mean_M_mean = mean(MEAN_norm(:, mid_idx),   2, 'omitnan');
mean_L_mean = mean(MEAN_norm(:, late_idx),  2, 'omitnan');

mean_E_resVol = mean(resid_STD(:, early_idx), 2, 'omitnan');
mean_M_resVol = mean(resid_STD(:, mid_idx),   2, 'omitnan');
mean_L_resVol = mean(resid_STD(:, late_idx),  2, 'omitnan');

%% 5. Logistic regression: predicting response = right

% Dependent variable: choose right (=1) vs left (=0)
resp_right = double(resp_vec == 1);   % 1 = right, 0 = left

% Design matrix X: 6 predictors
X = [mean_E_mean, mean_M_mean, mean_L_mean, ...
     mean_E_resVol, mean_M_resVol, mean_L_resVol];

y = resp_right;

% Remove any trials with NaN
valid_mask = all(~isnan(X), 2) & ~isnan(y);
X_valid    = X(valid_mask, :);
y_valid    = y(valid_mask);

fprintf('Using %d valid trials for logistic regression (response=right).\n', ...
    sum(valid_mask));

if isempty(X_valid)
    error('No valid trials left after removing NaNs.');
end

% glmfit automatically adds a constant term, so we use X_valid as is
[b, dev, stats] = glmfit(X_valid, y_valid, 'binomial', 'logit');

predictor_names = { ...
    'Intercept', ...
    'Mean_Early', ...
    'Mean_Mid', ...
    'Mean_Late', ...
    'ResVol_Early', ...
    'ResVol_Mid', ...
    'ResVol_Late'};

fprintf('\nLogistic regression results (predicting response = RIGHT = 1):\n');
for k = 1:numel(predictor_names)
    fprintf('%15s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        predictor_names{k}, b(k), stats.se(k), stats.p(k));
end

%% 6. Look at distributions of Early/Mid/Late residual volatility
%     for right vs left choices

idxR = (resp_right == 1) & valid_mask;
idxL = (resp_right == 0) & valid_mask;

figure;
subplot(3,1,1);
histogram(mean_E_resVol(idxR), 'Normalization','pdf'); hold on;
histogram(mean_E_resVol(idxL), 'Normalization','pdf');
legend({'Right choice','Left choice'});
xlabel('Residual volatility (Early)');
ylabel('Density');
title('Early residual volatility vs response');
grid on;

subplot(3,1,2);
histogram(mean_M_resVol(idxR), 'Normalization','pdf'); hold on;
histogram(mean_M_resVol(idxL), 'Normalization','pdf');
legend({'Right choice','Left choice'});
xlabel('Residual volatility (Mid)');
ylabel('Density');
title('Mid residual volatility vs response');
grid on;

subplot(3,1,3);
histogram(mean_L_resVol(idxR), 'Normalization','pdf'); hold on;
histogram(mean_L_resVol(idxL), 'Normalization','pdf');
legend({'Right choice','Left choice'});
xlabel('Residual volatility (Late)');
ylabel('Density');
title('Late residual volatility vs response');
grid on;


% Only use valid trials
r1 = corr(mean_E_resVol(valid_mask), abs(mean_E_mean(valid_mask)));
r2 = corr(mean_E_resVol(valid_mask), allStruct.rdm1_coh(valid_mask));
fprintf('Corr(Early resVol, |Early mean|) = %.8f\n', r1);
fprintf('Corr(Early resVol, coh)         = %.8f\n', r2);

% Simple correlation between Early residual volatility and choice
r3 = corr(mean_E_resVol(valid_mask), double(resp_right(valid_mask)));
fprintf('Corr(Early resVol, response=right) = %.8f\n', r3);
