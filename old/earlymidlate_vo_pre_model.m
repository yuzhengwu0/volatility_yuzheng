%% logit_mean_resVol_correct.m
% Goal:
% 1) From all.motion_energy, compute sliding-window mean & std for each trial
% 2) Put them onto a 0–1 normalized time axis
% 3) At each time bin, regress STD on |MEAN| and coherence, and get residual volatility
% 4) Split time into early / mid / late, get mean residual volatility for each part
% 5) Use logistic regression to see if these predictors can predict correct (0/1)

clear; clc;

%% 0. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell (signed evidence, right positive, left negative)
correct_vec   = allStruct.correct(:);      % 1 = correct, 0 = incorrect
nTrials       = numel(motion_energy);

fprintf('Loaded %d trials.\n', nTrials);

%% 1. For each trial, use sliding window to get mean & std (remove tail zeros)

winLen = 10;      % window length (10 frames)
tol    = 1e-12;   % small number to detect non-zero

evidence_strength   = cell(nTrials, 1);   % per trial: window mean
volatility_strength = cell(nTrials, 1);   % per trial: window std

for tr = 1:nTrials
    frames = motion_energy{tr};      % nFrames x 1
    trace  = frames(:)';             % 1 x nFrames (may have padding zeros at the end)
    
    % Find last non-zero frame (ignore tiny values)
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        % If all zeros, skip this trial
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    % Keep only frames with real signal
    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);
    
    % If trial is shorter than one window, skip it
    if nFrames < winLen
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    % Number of sliding windows in this trial
    nWin    = nFrames - winLen + 1;
    ev_mean = nan(1, nWin);
    ev_std  = nan(1, nWin);
    
    % Sliding window: move by 1 frame each time
    for w = 1:nWin
        segment    = trace_eff(w : w + winLen - 1);
        ev_mean(w) = mean(segment);   % average evidence in this window
        ev_std(w)  = std(segment);    % volatility (std) in this window
    end
    
    % Save results for this trial
    evidence_strength{tr}   = ev_mean;
    volatility_strength{tr} = ev_std;
end

%% 2. Put mean & std onto a common 0–1 time axis

nBins  = 40;                          % number of bins on normalized time axis
t_norm = linspace(0, 1, nBins);       % 0 = trial start, 1 = trial end

MEAN_norm = nan(nTrials, nBins);      % trials x time
STD_norm  = nan(nTrials, nBins);      % trials x time

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};    % window mean evidence for this trial
    sd_tr = volatility_strength{tr};  % window volatility for this trial
    
    if isempty(mu_tr) || isempty(sd_tr)
        % If no data (short or empty trial), skip
        continue;
    end
    
    % Make sure mean and std have same length (just in case)
    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);
    
    % Original time axis for this trial (0–1)
    t_orig = linspace(0, 1, nWin_tr);
    
    % Interpolate onto common time axis
    MEAN_norm(tr, :) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,  :) = interp1(t_orig, sd_tr, t_norm, 'linear');
end

%% 3. For each time bin, regress STD on |MEAN| and coherence, take residual STD (extra volatility)

coh_vec   = allStruct.rdm1_coh(:);    % coherence per trial
resid_STD = nan(size(STD_norm));      % trials x time

for b = 1:nBins
    y = STD_norm(:, b);            % volatility at this time bin (all trials)
    m = abs(MEAN_norm(:, b));      % |mean evidence| at this time bin
    c = coh_vec;                   % coherence (trial-level)
    
    % Keep trials with valid values
    mask = ~isnan(y) & ~isnan(m) & ~isnan(c);
    if sum(mask) < 10
        % Too few trials: skip this time bin
        continue;
    end
    
    % Design matrix: constant, |mean|, coherence
    X = [ones(sum(mask),1), m(mask), c(mask)];
    
    % Linear regression: y = b0 + b1*|mean| + b2*coherence + error
    beta = X \ y(mask);
    
    % Predicted STD from mean & coherence
    y_hat = X * beta;
    
    % Residual STD = actual STD - predicted STD
    tmp        = nan(size(y));
    tmp(mask)  = y(mask) - y_hat;   % extra volatility that mean & coherence cannot explain
    resid_STD(:, b) = tmp;
end

%% 4. Split time into Early / Mid / Late, get mean residual volatility

% Here we simply cut 40 bins into 3 parts (can be changed)
early_idx = 1:13;          % about 0   ~ 0.33
mid_idx   = 14:26;         % about 0.33~ 0.66
late_idx  = 27:40;         % about 0.66~ 1

% For each trial, compute average residual volatility in Early / Mid / Late

mean_E_resVol = mean(resid_STD(:, early_idx), 2, 'omitnan');
mean_M_resVol = mean(resid_STD(:, mid_idx),   2, 'omitnan');
mean_L_resVol = mean(resid_STD(:, late_idx),  2, 'omitnan');

% Build design matrix X: only Early/Mid/Late residual volatility
X = [mean_E_resVol, mean_M_resVol, mean_L_resVol];

y = correct_vec;   % 0/1 (0 = incorrect, 1 = correct)

% Remove trials with NaN in predictors or outcome
valid_mask = all(~isnan(X), 2) & ~isnan(y);
X_valid    = X(valid_mask, :);
y_valid    = y(valid_mask);

fprintf('Using %d valid trials for logistic regression.\n', sum(valid_mask));

%% 5. Logistic regression: correct ~ ResVol_Early + ResVol_Mid + ResVol_Late

if isempty(X_valid)
    error('No valid trials left after removing NaNs.');
end

% glmfit will add an intercept column automatically
[b, dev, stats] = glmfit(X_valid, y_valid, 'binomial', 'link', 'logit');

% Meaning of b:
% b(1): Intercept
% b(2): ResVol_Early
% b(3): ResVol_Mid
% b(4): ResVol_Late

predictor_names = { ...
    'Intercept', ...
    'ResVol_Early', ...
    'ResVol_Mid', ...
    'ResVol_Late'};

fprintf('\nLogistic regression results (predicting correct = 1):\n');
for k = 1:numel(predictor_names)
    fprintf('%15s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        predictor_names{k}, b(k), stats.se(k), stats.p(k));
end

%% 6. Optional: plot Early/Mid/Late residual volatility for correct vs incorrect (sanity check)

idxC = (correct_vec == 1) & valid_mask;   % correct trials
idxI = (correct_vec == 0) & valid_mask;   % incorrect trials

figure;
subplot(3,1,1);
histogram(mean_E_resVol(idxC), 'Normalization','pdf'); hold on;
histogram(mean_E_resVol(idxI), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Residual volatility (Early)');
ylabel('Density');
title('Early residual volatility distribution');
grid on;

subplot(3,1,2);
histogram(mean_M_resVol(idxC), 'Normalization','pdf'); hold on;
histogram(mean_M_resVol(idxI), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Residual volatility (Mid)');
ylabel('Density');
title('Mid residual volatility distribution');
grid on;

subplot(3,1,3);
histogram(mean_L_resVol(idxC), 'Normalization','pdf'); hold on;
histogram(mean_L_resVol(idxI), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Residual volatility (Late)');
ylabel('Density');
title('Late residual volatility distribution');
grid on;
