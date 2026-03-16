%% cumu_noise_kernel_reg.m
% Goal:
% 1) Use frame-by-frame motion_energy to compute real "cumulative evidence" and "cumulative noise":
%    - cumulative evidence: cumsum(evidence)
%    - cumulative noise: running variance of evidence (variance from the first frame up to now)
% 2) Normalize both to a 0–1 time axis
% 3) Plot the cumulative noise kernel (correct vs incorrect)
% 4) Run logistic regression: correct ~ CumEv_Late + CumNoise_Late

clear; clc;

%% 0. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

motion_energy = allStruct.motion_energy;   % 7584 x 1 cell (rightward = positive, leftward = negative)
correct_vec   = allStruct.correct(:);      % 1 = correct, 0 = incorrect

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. For each trial: compute cumulative evidence & cumulative noise (running variance)

tol = 1e-12;   % Threshold to detect tail zeros (so tiny numbers are not treated as real data)

cumEv_cell    = cell(nTrials, 1);   % cumulative evidence for each trial
cumNoise_cell = cell(nTrials, 1);   % cumulative noise (running variance) for each trial

for tr = 1:nTrials
    frames = motion_energy{tr};   % nFrames x 1
    trace  = frames(:)';          % 1 x nFrames, may have padding zeros at the end
    
    % Remove padding zeros at the end: keep up to the last non-zero point
    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        cumEv_cell{tr}    = [];
        cumNoise_cell{tr} = [];
        continue;
    end
    
    trace_eff = trace(1:last_nz);     % effective part of the trace
    nF        = numel(trace_eff);
    
    % ---- cumulative evidence: sum from the start ----
    cumEv = cumsum(trace_eff);        % sum of frames 1..k
    
    % ---- cumulative noise: running variance ----
    % Variance Var(X_1..X_k) = E[X^2] - (E[X])^2
    nVec     = 1:nF;
    cum_sum  = cumsum(trace_eff);         % Σ X
    cum_sq   = cumsum(trace_eff.^2);      % Σ X^2
    mean_run = cum_sum ./ nVec;           % running mean for frames 1..k
    var_run  = cum_sq ./ nVec - mean_run.^2;  % running variance
    
    % Small numerical errors may give tiny negative values → clamp them to 0
    var_run(var_run < 0) = 0;
    
    cumEv_cell{tr}    = cumEv;
    cumNoise_cell{tr} = var_run;    % here we define "noise" as this running variance
end

%% 2. Normalize cumulative evidence / noise to a common 0–1 time axis

nBins  = 40;                       % number of time bins after normalization
t_norm = linspace(0, 1, nBins);    % 0 = trial start, 1 = trial end

CUMEV_norm    = nan(nTrials, nBins);   % trials x time bins
CUMNOISE_norm = nan(nTrials, nBins);   % trials x time bins

for tr = 1:nTrials
    ce_tr = cumEv_cell{tr};
    cn_tr = cumNoise_cell{tr};
    
    if isempty(ce_tr) || isempty(cn_tr)
        continue;
    end
    
    nF_tr = min(numel(ce_tr), numel(cn_tr));
    ce_tr = ce_tr(1:nF_tr);
    cn_tr = cn_tr(1:nF_tr);
    
    t_orig = linspace(0, 1, nF_tr);   % original time points for this trial
    
    CUMEV_norm(tr, :)    = interp1(t_orig, ce_tr, t_norm, 'linear');
    CUMNOISE_norm(tr, :) = interp1(t_orig, cn_tr, t_norm, 'linear');
end

%% 4. Only take Late cumulative summary (avoid Early/Mid/Late collinearity)

late_idx  = 27:40;   % ~ 0.66–1 of the trial

% —— Late summary of cumulative EVIDENCE ——
CumEv_L = mean(CUMEV_norm(:, late_idx), 2, 'omitnan');

% —— Late summary of cumulative NOISE ——
CumN_L  = mean(CUMNOISE_norm(:, late_idx), 2, 'omitnan');

%% 5. Logistic regression: correct ~ CumEv_Late + CumNoise_Late

X = [CumEv_L, CumN_L];   % use one time window only, greatly reduces collinearity

y = correct_vec;

valid_mask = all(~isnan(X), 2) & ~isnan(y);
X_valid    = X(valid_mask, :);
y_valid    = y(valid_mask);

fprintf('Using %d valid trials for logistic regression (late cumulative model).\n', ...
    sum(valid_mask));

if isempty(X_valid)
    error('No valid trials left after removing NaNs.');
end

% (Optional) z-score predictors so beta sizes are easier to compare
X_z = zscore(X_valid);   % each column: subtract mean and divide by std

[b, dev, stats] = glmfit(X_z, y_valid, 'binomial', 'logit');

predictor_names = { ...
    'Intercept', ...
    'CumEv_Late_z', ...
    'CumNoise_Late_z'};

fprintf('\nLogistic regression results (predicting correct = 1):\n');
for k = 1:numel(predictor_names)
    fprintf('%15s: beta = %+ .4e,  SE = %.4e,  p = %.4g\n', ...
        predictor_names{k}, b(k), stats.se(k), stats.p(k));
end

%% 6. (Optional) Look only at Late cumulative noise for correct vs incorrect

idxC_valid = (correct_vec == 1) & valid_mask;
idxI_valid = (correct_vec == 0) & valid_mask;

figure;
histogram(CumN_L(idxC_valid), 'Normalization','pdf'); hold on;
histogram(CumN_L(idxI_valid), 'Normalization','pdf');
legend({'Correct','Incorrect'});
xlabel('Cumulative noise (Late)');
ylabel('Density');
title('Late cumulative noise vs correctness');
grid on;

% On valid trials, regress CumNoise_L on |CumEv_L| and take residual as "pure noise"
absEv_L = abs(CumEv_L(valid_mask));
noise_L = CumN_L(valid_mask);

X_ev   = [ones(size(absEv_L)), absEv_L];
beta_ev   = X_ev \ noise_L;
noise_resid = noise_L - X_ev * beta_ev;  % remove the part of noise that can be explained by |evidence|

% Then in logistic regression, use CumEv_L and noise_resid as predictors
X_valid = [absEv_L, noise_resid];
X_z     = zscore(X_valid);
