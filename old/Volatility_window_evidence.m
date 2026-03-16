%% compute_sliding_volatility.m
% We use all.motion_energy (each trial is nFrames x 1 double, right = positive, left = negative).
% For each trial, we use a sliding window of length 10 frames (step = 1 frame).
% In each window, we compute evidence volatility = standard deviation (std) of evidence.
% Then we build a Correct vs Incorrect volatility kernel in normalized time.

clear; clc;

%% 1. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';

% Only load the variable "all" from the .mat file
tmp = load(data_path, 'all');
allStruct = tmp.all;                       % 1x1 struct

% Signed evidence (right positive, left negative)
motion_energy = allStruct.motion_energy;   % 7584 x 1 cell, each cell: nFrames x 1 double

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 2. Sliding window parameters
winLen = 10;                               % Each window has 10 frames
volatility_strength = cell(nTrials, 1);    % For each trial: volatility (std) for each window

%% 3. For each trial, do sliding window (remove zero tail first, then compute std)
for tr = 1:nTrials
    frames = motion_energy{tr};    % nFrames x 1 double for this trial
    trace  = frames(:)';           % Make it a 1 x nFrames row vector (may include padding zeros)
    
    % --- Remove tail zeros: keep only up to the last non-zero frame ---
    tol = 1e-12;
    last_nz = find(abs(trace) > tol, 1, 'last');
    
    if isempty(last_nz)
        % This trial has no non-zero evidence, set empty and skip
        volatility_strength{tr} = [];
        continue;
    end
    
    trace_eff = trace(1:last_nz);  % Effective part (frames with real evidence)
    nFrames   = numel(trace_eff);
    
    if nFrames < winLen
        % Not enough frames to form one full window
        volatility_strength{tr} = [];
        continue;
    end
    
    nWin = nFrames - winLen + 1;   % Number of sliding windows in this trial
    ev_std = nan(1, nWin);         % Volatility (std) for each window in this trial
    
    for w = 1:nWin
        % Current window: 10 frames of evidence
        segment = trace_eff(w : w + winLen - 1);
        % Volatility = standard deviation of evidence in this window
        ev_std(w) = std(segment);
    end
    
    % Save the volatility time series for this trial
    volatility_strength{tr} = ev_std;
end

%% 4. Print sliding-window volatility for the first 3 trials
nPrint = min(3, nTrials);
for tr = 1:nPrint
    vol_tr = volatility_strength{tr};
    fprintf('\nTrial %d: %d sliding windows (volatility = std of evidence):\n', ...
        tr, numel(vol_tr));
    disp(vol_tr);
end

%% 5. Plot one trial: evidence & volatility (just to see the shape)

tr_plot = 2;   % Which trial to plot (you can change this)

frames = motion_energy{tr_plot};    % Original signed evidence for this trial
trace  = frames(:)';                % 1 x nFrames (may include tail zeros)

% Again remove tail zeros so it matches the volatility calculation
tol = 1e-12;
last_nz = find(abs(trace) > tol, 1, 'last');
if isempty(last_nz)
    warning('Trial %d is all zeros.', tr_plot);
else
    trace_eff = trace(1:last_nz);          % Effective evidence
    nFrames_eff = numel(trace_eff);        % Number of effective frames
    
    vol_tr = volatility_strength{tr_plot}; % Volatility (std) for each window in this trial
    nWin   = numel(vol_tr);
    % Window center frame index for plotting (rough position of each window)
    centerFrames = (1:nWin) + (winLen - 1)/2;
    
    figure;
    subplot(2,1,1);
    plot(1:nFrames_eff, trace_eff, '-');
    xlabel('Frame index');
    ylabel('Evidence (right - left)');
    title(sprintf('Trial %d: effective frame-by-frame evidence', tr_plot));
    grid on;
    
    subplot(2,1,2);
    plot(centerFrames, vol_tr, '-o');
    xlabel('Frame index (window center)');
    ylabel('Volatility (std of evidence)');
    title(sprintf('Trial %d: %d-frame sliding-window volatility (step = 1)', ...
        tr_plot, winLen));
    grid on;
end

%% 6. Normalized-time VOLATILITY kernel (Correct vs Incorrect)

% correct: 1 = correct trial, 0 = incorrect trial
correct_vec = allStruct.correct(:);        % Make sure it is a column vector

% Number of windows in each trial (based on volatility_strength)
winLens_vol   = cellfun(@numel, volatility_strength);
valid_idx_vol = (winLens_vol > 0);         % Trials that have at least 1 window

% Indices of valid correct and incorrect trials
correct_trials_vol   = find(correct_vec == 1 & valid_idx_vol);
incorrect_trials_vol = find(correct_vec == 0 & valid_idx_vol);

nCorrect_vol   = numel(correct_trials_vol);
nIncorrect_vol = numel(incorrect_trials_vol);

fprintf('\n[Volatility] Number of correct trials (valid): %d\n',   nCorrect_vol);
fprintf('[Volatility] Number of incorrect trials (valid): %d\n', nIncorrect_vol);

if nCorrect_vol == 0 || nIncorrect_vol == 0
    warning('Not enough correct/incorrect trials to compute volatility kernel.');
else
    % Number of normalized time points (we resample each trial to 40 time bins)
    nBins_vol  = 40;
    t_norm_vol = linspace(0, 1, nBins_vol);   % 0 = trial start, 1 = trial end
    
    % Volatility on normalized time axis: rows = trials, cols = time bins
    VOL_correct_norm   = nan(nCorrect_vol,   nBins_vol);
    VOL_incorrect_norm = nan(nIncorrect_vol, nBins_vol);
    
    % ---- Correct trials: interpolate each trial's volatility to 0–1 ----
    for i = 1:nCorrect_vol
        tr     = correct_trials_vol(i);
        vol_tr = volatility_strength{tr};     % Volatility (std) time series for this trial
        
        if isempty(vol_tr)
            continue;
        end
        
        nWin_tr = numel(vol_tr);             % Number of windows in this trial
        % Original time axis for this trial, mapped to 0–1 (start to end)
        t_orig  = linspace(0, 1, nWin_tr);
        
        % Interpolate this trial's volatility onto the common normalized time axis
        VOL_correct_norm(i, :) = interp1(t_orig, vol_tr, t_norm_vol, 'linear');
    end
    
    % ---- Incorrect trials: same interpolation ----
    for i = 1:nIncorrect_vol
        tr     = incorrect_trials_vol(i);
        vol_tr = volatility_strength{tr};
        
        if isempty(vol_tr)
            continue;
        end
        
        nWin_tr = numel(vol_tr);
        t_orig  = linspace(0, 1, nWin_tr);
        
        VOL_incorrect_norm(i, :) = interp1(t_orig, vol_tr, t_norm_vol, 'linear');
    end
    
    % For each normalized time bin, average across trials (ignore NaN)
    mean_VOL_correct_norm   = mean(VOL_correct_norm,   1, 'omitnan');
    mean_VOL_incorrect_norm = mean(VOL_incorrect_norm, 1, 'omitnan');
    
    % Volatility kernel: Correct minus Incorrect at each time bin
    kernel_vol_norm = mean_VOL_correct_norm - mean_VOL_incorrect_norm;
    
    % Plot results
    figure;
    subplot(2,1,1);
    plot(t_norm_vol, mean_VOL_correct_norm,   '-g', 'LineWidth', 1.2); hold on;
    plot(t_norm_vol, mean_VOL_incorrect_norm, '-m', 'LineWidth', 1.2);
    xlabel('Normalized time within trial (0 = start, 1 = end)');
    ylabel('Volatility (std of evidence)');
    legend({'Correct trials','Incorrect trials'}, 'Location','best');
    title('Normalized-time VOLATILITY (Correct vs Incorrect)');
    grid on;
    
    subplot(2,1,2);
    plot(t_norm_vol, kernel_vol_norm, '-k', 'LineWidth', 1.2); hold on;
    yline(0, '--');
    xlabel('Normalized time within trial (0 = start, 1 = end)');
    ylabel('Volatility difference (Correct - Incorrect)');
    title('VOLATILITY accuracy kernel over time');
    grid on;
end
