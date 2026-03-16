%% compute_sliding_evidence_L.m
% Read all.motion_energy_L (each trial is nFrames x 1 double, only LEFT evidence)
% For each trial, do sliding window with length = 10 frames (step = 1 frame)
% For each window, compute "left evidence" strength
% Then make some simple plots

clear; clc;

%% 1. Read data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';

% Only load variable "all"
tmp = load(data_path, 'all');
allStruct = tmp.all;                       % 1x1 struct

% From "all", take the trace that has only LEFT evidence
motion_energy_L = allStruct.motion_energy_L;   % 7584 x 1 cell, each is nFrames x 1 double

nTrials = numel(motion_energy_L);
fprintf('Loaded %d trials.\n', nTrials);

%% 2. Sliding window settings
winLen = 10;                        % Each window has 10 frames
evidence_L_strength = cell(nTrials, 1);   % For each trial, one sequence of "LEFT evidence" windows

%% 3. For each trial, do sliding window (only LEFT evidence, remove zeros at the end)
for tr = 1:nTrials
    % One element is nFrames x 1 double, this is LEFT evidence for each frame in this trial
    frames_L = motion_energy_L{tr};     % nFrames x 1 double
    
    % Change to row vector, easy to index
    trace_L = frames_L(:)';             % 1 x nFrames (maybe has padding zeros)
    
    % --- Keep only real data: cut at last non-zero frame ---
    tol = 1e-12;
    last_nz = find(abs(trace_L) > tol, 1, 'last');
    
    if isempty(last_nz)
        % This trial is all zeros
        evidence_L_strength{tr} = [];
        continue;
    end
    
    trace_eff = trace_L(1:last_nz);     % Effective part
    nFrames = numel(trace_eff);
    
    if nFrames < winLen
        % Not enough frames for 10-frame window
        evidence_L_strength{tr} = [];
        continue;
    end
    
    nWin = nFrames - winLen + 1;        % Number of windows in this trial
    evL = nan(1, nWin);
    
    for w = 1:nWin
        % Take 10 frames for this window (LEFT evidence)
        segment_L = trace_eff(w : w + winLen - 1);
        
        % ===== Define "LEFT evidence" strength here =====
        % Here we use mean of 10-frame LEFT evidence
        evL(w) = mean(segment_L);
        
        % If you want volatility, you can use: evL(w) = std(segment_L);
    end
    
    evidence_L_strength{tr} = evL;
end

%% 4. Print first 3 trials' sliding window results (LEFT evidence)
nPrint = min(3, nTrials);
for tr = 1:nPrint
    evL = evidence_L_strength{tr};
    fprintf('\nTrial %d: %d sliding windows (left evidence):\n', tr, numel(evL));
    disp(evL);
end

%% 5. Plot: one trial's LEFT evidence & sliding window changes

tr_plot = 2;   % Which trial to show, you can change

% This trial's original frame-by-frame LEFT evidence
frames_L = motion_energy_L{tr_plot};     % nFrames x 1 double
trace_L_full  = frames_L(:)';            % 1 x nFrames

% This trial's sliding-window LEFT evidence
evL      = evidence_L_strength{tr_plot}; % 1 x nWin
nFrames_full  = numel(trace_L_full);
nWin     = numel(evL);

% To match frame index, give each window a "center" frame
centerFrames = (1:nWin) + (winLen-1)/2;  % winLen=10 → shift by 4.5

figure;
subplot(2,1,1);
plot(1:nFrames_full, trace_L_full, '-');
xlabel('Frame index');
ylabel('Left-motion evidence');
title(sprintf('Trial %d: Original frame-by-frame evidence to left', tr_plot));
grid on;

subplot(2,1,2);
plot(centerFrames, evL, '-o');
xlabel('Frame index (window center)');
ylabel('Sliding-window left evidence');
title(sprintf('Trial %d: %d-frame left sliding window (step = 1 frame)', tr_plot, winLen));
grid on;

%% 6. Average sliding-window LEFT evidence for trials with LEFT response (absolute frame index)

% req_resp: 1 = right, 2 = left
req_resp = allStruct.req_resp(:);

left_idx    = (req_resp == 2);
left_trials = find(left_idx);
nLeft       = numel(left_trials);

fprintf('\nNumber of trials with left response: %d\n', nLeft);

if nLeft == 0
    warning('No trials with left response (req_resp == 2). Cannot compute average.');
else
    % Find shortest window length among all LEFT-choice trials
    minWin_L = inf;
    for i = 1:nLeft
        tr = left_trials(i);
        ev_tr = evidence_L_strength{tr};
        if ~isempty(ev_tr)
            minWin_L = min(minWin_L, numel(ev_tr));
        end
    end
    
    if isinf(minWin_L)
        warning('All evidence_L_strength entries for left-choice trials are empty.');
    else
        % Put into matrix: trial x window
        EV_left = nan(nLeft, minWin_L);
        for i = 1:nLeft
            tr    = left_trials(i);
            ev_tr = evidence_L_strength{tr};
            EV_left(i, :) = ev_tr(1:minWin_L);
        end
        
        % Average across trials
        mean_EV_left = mean(EV_left, 1, 'omitnan');
        
        % Window centers (same winLen)
        centerFrames_left = (1:minWin_L) + (winLen - 1)/2;
        
    end
end

%% 7. Normalized-time average LEFT evidence for trials with LEFT response (0–1 time)

if nLeft > 0 && exist('minWin_L', 'var') && ~isinf(minWin_L)
    
    nBinsL  = 40;                           % Number of time bins after normalization
    t_normL = linspace(0, 1, nBinsL);       % 0 = trial start, 1 = trial end
    
    EV_left_norm = nan(nLeft, nBinsL);
    
    for i = 1:nLeft
        tr = left_trials(i);
        ev_tr = evidence_L_strength{tr};   % This trial's LEFT evidence window sequence
        
        if isempty(ev_tr)
            continue;
        end
        
        nWin_tr_L = numel(ev_tr);
        % Original window times mapped to 0–1
        t_orig_L  = linspace(0, 1, nWin_tr_L);
        
        % Interpolate onto common 0–1 time axis
        EV_left_norm(i, :) = interp1(t_orig_L, ev_tr, t_normL, 'linear');
    end
    
    % Average across trials
    mean_EV_left_norm = mean(EV_left_norm, 1, 'omitnan');
    
    figure;
    plot(t_normL, mean_EV_left_norm, '-o');
    xlabel('Normalized time in trial (0 = start, 1 = end)');
    ylabel('Mean leftward evidence');
    title(sprintf(['Normalized-time sliding-window LEFT evidence\n' ...
                   'for trials with left response (N = %d, %d bins)'], ...
                   nLeft, nBinsL));
    grid on;
end
