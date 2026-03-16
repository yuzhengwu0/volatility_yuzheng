%% compute_sliding_evidence_R.m
% Read all.motion_energy_R (each trial is nFrames x 1 double, only RIGHT evidence)
% For each trial, use a sliding window of 10 frames (step = 1 frame)
% For each window, compute "right evidence" strength
% Then print first 3 trials and make some plots

clear; clc;

%% 1. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';

% Only load variable "all"
tmp = load(data_path, 'all');
allStruct = tmp.all;   % 1x1 struct

% From "all", take the traces that have only RIGHT evidence
motion_energy_R = allStruct.motion_energy_R;   % 7584 x 1 cell, each is nFrames x 1 double

nTrials = numel(motion_energy_R);
fprintf('Loaded %d trials.\n', nTrials);

%% 2. Sliding window settings
winLen = 10;                             % Each window has 10 frames
evidence_R_strength = cell(nTrials, 1);  % For each trial, one vector of windowed RIGHT evidence

%% 3. Sliding window for each trial (RIGHT evidence only)
for tr = 1:nTrials
    % One element is nFrames x 1 double, RIGHT evidence for each frame in this trial
    frames_R = motion_energy_R{tr};     % nFrames x 1 double
    
    % Change to row vector
    trace_R = frames_R(:)';             % 1 x nFrames (may include padding zeros)
    
    % ---- Keep only real data: cut at last non-zero frame ----
    % Use a small threshold, so tiny numbers are not treated as zero by mistake
    tol = 1e-12;
    last_nz = find(abs(trace_R) > tol, 1, 'last');  % index of last non-zero
    
    if isempty(last_nz)
        % This trial is all zeros, set empty and skip
        evidence_R_strength{tr} = [];
        continue;
    end
    
    trace_eff = trace_R(1:last_nz);     % Keep only frames 1:last_nz
    nFrames = numel(trace_eff);
    
    if nFrames < winLen
        % Not enough frames for a 10-frame window
        evidence_R_strength{tr} = [];
        continue;
    end
    
    nWin = nFrames - winLen + 1;        % Number of windows in this trial
    evR = nan(1, nWin);
    
    for w = 1:nWin
        % Take 10 frames for this window (real data part)
        segment_R = trace_eff(w : w + winLen - 1);
        
        % Use mean RIGHT evidence over these 10 frames
        evR(w) = mean(segment_R);
    end
    
    % Save sliding-window RIGHT evidence for this trial
    evidence_R_strength{tr} = evR;
end

%% 4. Print sliding-window results (RIGHT evidence) for first 3 trials
nPrint = min(3, nTrials);
for tr = 1:nPrint
    evR = evidence_R_strength{tr};
    fprintf('\nTrial %d: %d sliding windows (right evidence):\n', tr, numel(evR));
    disp(evR);
end

%% 5. Plot one trial: frame-by-frame RIGHT evidence & sliding-window RIGHT evidence

tr_plot = 2;   % Choose which trial to show

% Original frame-by-frame RIGHT evidence for this trial
frames_R = motion_energy_R{tr_plot};     % nFrames x 1 double
trace_R  = frames_R(:)';                 % 1 x nFrames

% Sliding-window RIGHT evidence for this trial
evR      = evidence_R_strength{tr_plot}; % 1 x nWin
nFrames  = numel(trace_R);
nWin     = numel(evR);

% Give each window a center frame index
centerFrames = (1:nWin) + (winLen - 1)/2;  % winLen=10 → shift by 4.5

figure;
subplot(2,1,1);
plot(1:nFrames, trace_R, '-');
xlabel('Frame index');
ylabel('Right-motion evidence');
title(sprintf('Trial %d: Original frame-by-frame right evidence', tr_plot));
grid on;

subplot(2,1,2);
plot(centerFrames, evR, '-o');
xlabel('Frame index (window center)');
ylabel('Sliding-window right evidence');
title(sprintf('Trial %d: %d-frame right sliding window (step = 1 frame)', tr_plot, winLen));
grid on;

%% 6. Average sliding-window RIGHT evidence for trials with RIGHT response (absolute frame index)

% Subject responses: 1 = right, 2 = left
req_resp = allStruct.req_resp;   % nTrials x 1 or 1 x nTrials
req_resp = req_resp(:);          % force column vector

% Find trials where subject chose RIGHT
right_idx    = (req_resp == 1);
right_trials = find(right_idx);
nRight       = numel(right_trials);

fprintf('\nNumber of trials with right response: %d\n', nRight);

if nRight == 0
    warning('No trials with right response (req_resp == 1). Cannot compute average.');
else
    % Find smallest window length among RIGHT-choice trials
    minWin = inf;
    for i = 1:nRight
        tr = right_trials(i);
        ev_tr = evidence_R_strength{tr};
        if ~isempty(ev_tr)
            minWin = min(minWin, numel(ev_tr));
        end
    end
    
    if isinf(minWin)
        warning('All evidence_R_strength entries for right-choice trials are empty.');
    else
        % Put RIGHT-choice trials into a matrix: trials x windows
        EV_right = nan(nRight, minWin);
        for i = 1:nRight
            tr    = right_trials(i);
            ev_tr = evidence_R_strength{tr};
            EV_right(i, :) = ev_tr(1:minWin);
        end
        
        % Average across trials (ignore NaNs)
        mean_EV_right = mean(EV_right, 1, 'omitnan');
        
        % Window center frames for these minWin windows
        centerFrames_right = (1:minWin) + (winLen - 1)/2;
        
    end
end

%% 7. Normalized-time average RIGHT evidence for trials with RIGHT response (0–1 time)

% Only do this if we have RIGHT-choice trials
if nRight > 0
    
    % Number of time bins after normalization, for example 40 (you can change)
    nBins = 40;
    t_norm = linspace(0, 1, nBins);   % 0 = trial start, 1 = trial end
    
    % Store normalized RIGHT evidence for each RIGHT-choice trial
    EV_right_norm = nan(nRight, nBins);
    
    for i = 1:nRight
        tr = right_trials(i);
        ev_tr = evidence_R_strength{tr};   % Sliding-window sequence for this trial
        
        if isempty(ev_tr)
            continue;
        end
        
        nWin_tr = numel(ev_tr);
        
        % Original time axis for this trial, mapped 0–1
        t_orig = linspace(0, 1, nWin_tr);
        
        % Interpolate onto common normalized time axis t_norm
        EV_right_norm(i, :) = interp1(t_orig, ev_tr, t_norm, 'linear');
    end
    
    % Average across trials on normalized time axis
    mean_EV_right_norm = mean(EV_right_norm, 1, 'omitnan');
    
    % Plot normalized-time RIGHT evidence
    figure;
    plot(t_norm, mean_EV_right_norm, '-o');
    xlabel('Normalized time within trial (0 = start, 1 = end)');
    ylabel('Mean rightward evidence');
    title(sprintf(['Normalized-time sliding-window right evidence\n' ...
                   'for trials with right response (N = %d, %d bins)'], ...
                   nRight, nBins));
    grid on;
end
