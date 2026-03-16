%% logit_resVol_confBias.m
% Goal:
% 1) Read frame-by-frame signed evidence from all.motion_energy (right = +, left = -)
% 2) Use 10-frame sliding window to get mean evidence and volatility (std) in each window
% 3) Stretch mean and std into a 0–1 time axis (40 time points)
% 4) At each time bin, do STD ~ |MEAN| + coherence, and take residual STD = extra volatility
% 5) Cut time into Early / Mid / Late, and use these three residual volatility values as predictors
% 6) Do logistic regression: see if residual volatility makes a trial more likely to be "high confidence"
%
% Note: we cut confidence into high / low by the median.
%       We ask: does volatility increase the chance of a "high confidence" response (confidence bias)?

clear; clc;

%% 0. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell (difference evidence, right positive, left negative)
nTrials       = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 0.1 Load confidence
% !!! Change this to your real confidence variable name !!!
% For example: allStruct.conf / allStruct.confidence / allStruct.rating etc.
conf_vec = allStruct.confidence(:);   % <--- change this line if the name is not "confidence"

% Ignore NaN and check confidence range
fprintf('Confidence range (ignoring NaNs): [%.2f, %.2f]\n', ...
        min(conf_vec(~isnan(conf_vec))), max(conf_vec(~isnan(conf_vec))));

% Define "high confidence": above the median of all trials
medConf   = median(conf_vec(~isnan(conf_vec)));
high_conf = conf_vec > medConf;   % 1 = high, 0 = low

fprintf('Median confidence = %.2f\n', medConf);
fprintf('High-confidence trials: %d / %d (%.1f%%)\n', ...
        sum(high_conf==1 & ~isnan(high_conf)), ...
        sum(~isnan(high_conf)), ...
        100 * sum(high_conf==1 & ~isnan(high_conf)) / sum(~isnan(high_conf)));

%% 0.2 Load coherence (stimulus difficulty)
% !!! Also change rdm1_coh to your real coherence variable name !!!
coh_vec = allStruct.rdm1_coh(:);   % one coherence per trial
fprintf('Coherence range (ignoring NaNs): [%.3f, %.3f]\n', ...
        min(coh_vec(~isnan(coh_vec))), max(coh_vec(~isnan(coh_vec))));

%% 1. For each trial, do sliding window and get mean & std (cut zeros at the end)

winLen = 10;
tol    = 1e-12;

evidence_strength   = cell(nTrials, 1);   % for each trial: window mean
volatility_strength = cell(nTrials, 1);   % for each trial: window std

for tr = 1:nTrials
    frames = motion_energy{tr};   % nFrames x 1
    trace  = frames(:)';          % 1 x nFrames (may have padding zeros)
    
    % Cut zeros at the end: keep up to the last non-zero frame
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);
    
    % If too few frames, skip this trial
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
        ev_mean(w) = mean(segment);   % mean evidence in this window
        ev_std(w)  = std(segment);    % volatility in this window
    end
    
    evidence_strength{tr}   = ev_mean;
    volatility_strength{tr} = ev_std;
end

%% 2. Stretch mean & std to a common 0–1 time axis (40 time points)

nBins  = 40;                       % number of bins on normalized time
t_norm = linspace(0, 1, nBins);    % 0 = start of trial, 1 = end of trial

MEAN_norm = nan(nTrials, nBins);   % trial x time
STD_norm  = nan(nTrials, nBins);   % trial x time

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};
    
    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end
    
    % Safety: use the shorter length of the two
    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);
    
    % Original time axis for this trial (0–1)
    t_orig = linspace(0, 1, nWin_tr);
    
    MEAN_norm(tr, :) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,  :) = interp1(t_orig, sd_tr, t_norm, 'linear');
end

%% 3. At each time bin, do STD ~ |MEAN| + coherence, and get residual STD (extra volatility)

resid_STD = nan(size(STD_norm));   % trial x time

for b = 1:nBins
    y = STD_norm(:, b);            % volatility at this time bin
    m = abs(MEAN_norm(:, b));      % |mean evidence| at this time bin
    c = coh_vec;                   % coherence for this trial (same for whole trial)
    
    % Keep only trials with no NaN in y, m, c
    mask = ~isnan(y) & ~isnan(m) & ~isnan(c);
    if sum(mask) < 10
        continue;
    end
    
    % Design matrix: [constant, |mean|, coherence]
    Xb    = [ones(sum(mask),1), m(mask), c(mask)];
    betab = Xb \ y(mask);                   % fit y = b0 + b1*|mean| + b2*coh
    
    % Predicted "normal" std given |mean| and coherence
    y_hat = Xb * betab;
    
    % residual = real std - normal std
    tmp       = nan(size(y));
    tmp(mask) = y(mask) - y_hat;
    resid_STD(:, b) = tmp;                  % extra volatility (beyond mean + coh)
end

%% 4. Cut time into Early / Mid / Late,
%   and take residual volatility as trial-level predictors

early_idx = 1:13;       % about 0 ~ 0.33
mid_idx   = 14:26;      % about 0.33 ~ 0.66
late_idx  = 27:40;      % about 0.66 ~ 1

% 4.1 residual volatility
ResVol_E = mean(resid_STD(:, early_idx), 2, 'omitnan');
ResVol_M = mean(resid_STD(:, mid_idx),   2, 'omitnan');
ResVol_L = mean(resid_STD(:, late_idx),  2, 'omitnan');

% 4.2 (optional) mean evidence by time period
%MeanAbs_E = mean(abs(MEAN_norm(:, early_idx)), 2, 'omitnan');
%MeanAbs_L = mean(abs(MEAN_norm(:, late_idx)),  2, 'omitnan');

%% 5. Build logistic regression:
%   high_conf ~ [ResVol_E / ResVol_M / ResVol_L]

y = high_conf;   % 0 = low confidence, 1 = high confidence

% If you only want correct trials, use:
% correct_mask = (allStruct.correct(:) == 1);
correct_mask = true(nTrials,1);

% Put all trial-level predictors into one matrix
predMat = [ResVol_E, ResVol_M, ResVol_L];

% Drop trials with NaN in predictors or y
valid_mask = all(~isnan(predMat), 2) & ~isnan(y) & correct_mask;

X_valid = predMat(valid_mask, :);
y_valid = y(valid_mask);

fprintf('Using %d valid trials for logistic regression (confidence bias).\n', ...
        sum(valid_mask));

if isempty(X_valid)
    error('No valid trials left after removing NaNs.');
end

% Do z-score for predictors in logistic regression
X_z = zscore(X_valid);   % subtract mean and divide by std for each column

[b, dev, stats] = glmfit(X_z, y_valid, 'binomial', 'link', 'logit');

pred_names = { ...
    'Intercept', ...
    'ResVol_Early', ...
    'ResVol_Mid', ...
    'ResVol_Late'};

fprintf('\nLogistic regression: predicting HIGH confidence (1 = high, 0 = low)\n');
for k = 1:numel(pred_names)
    fprintf('%15s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        pred_names{k}, b(k), stats.se(k), stats.p(k));
end

%% 6. Plot distributions of residual volatility
%   Compare High vs Low confidence trials (use raw values, not z-scored)

idxHigh = (y == 1) & valid_mask;
idxLow  = (y == 0) & valid_mask;

figure;
subplot(3,1,1);
histogram(ResVol_E(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_E(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('Residual volatility (Early)');
ylabel('Density');
title('Early residual volatility vs confidence');
grid on;

subplot(3,1,2);
histogram(ResVol_M(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_M(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('Residual volatility (Mid)');
ylabel('Density');
title('Mid residual volatility vs confidence');
grid on;

subplot(3,1,3);
histogram(ResVol_L(idxHigh), 'Normalization','pdf'); hold on;
histogram(ResVol_L(idxLow),  'Normalization','pdf');
legend({'High conf','Low conf'});
xlabel('Residual volatility (Late)');
ylabel('Density');
title('Late residual volatility vs confidence');
grid on;
