%% compute_sliding_evidence_cell.m
% Read motion_energy from a two-level cell / struct
% For each trial, do a sliding window of 10 frames (step = 1 frame)
% For each window, compute evidence strength (mean motion energy)

clear; clc;

%% 1. Load data
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
% Only load variable "all"
tmp = load(data_path, 'all');

% Get motion_energy from the struct "all"
allStruct     = tmp.all;                    % 1x1 struct
motion_energy = allStruct.motion_energy;    % 7584 x 1 cell

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 2. Sliding window parameters
winLen = 10;  % Each window is 10 frames long

% Use a cell to store results for each trial,
% because each trial can have a different number of frames
% (and a different number of valid frames)
evidence_strength = cell(nTrials, 1);


%% 3. For each trial, do sliding window (cut off tail zeros first, then compute)
for tr = 1:nTrials
    % Each element is an nFrames x 1 double.
    % Each frame is (right - left) evidence
    frames = motion_energy{tr};        % nFrames x 1 double
    trace  = frames(:)';               % 1 x nFrames, includes padding zeros
    
    % ---- Remove tail padding zeros: keep only up to the last non-zero ----
    tol = 1e-12;                       % Small threshold, to avoid tiny numbers being treated as 0
    last_nz = find(abs(trace) > tol, 1, 'last');   % Index of the last non-zero
    
    if isempty(last_nz)
        % The whole trial is zero, skip this trial
        evidence_strength{tr} = [];
        continue;
    end
    
    trace_eff = trace(1:last_nz);      % Valid part (from frame 1 to last non-zero)
    nFrames   = numel(trace_eff);
    
    if nFrames < winLen
        % Not enough valid frames for a 10-frame window
        evidence_strength{tr} = [];
        continue;
    end
    
    % Number of windows for this trial (using only the valid part)
    nWin = nFrames - winLen + 1;
    ev   = nan(1, nWin);
    
    for w = 1:nWin
        % Current window: 10 valid frames
        segment = trace_eff(w : w + winLen - 1);
        
        % Use the mean of 10 frames of motion_energy as evidence
        % (difference evidence, right positive, left negative)
        ev(w) = mean(segment);
    end
    
    % Save window evidence for this trial
    evidence_strength{tr} = ev;
end

%% 5. Visualization: single trial evidence & sliding window change

tr_plot = 2;   % Which trial to show, you can change this

% Original frame-by-frame motion energy for this trial (with padding)
frames = motion_energy{tr_plot};    % nFrames x 1 double
trace  = frames(:)';                % 1 x nFrames

% Sliding window result for this trial (only up to last non-zero)
ev = evidence_strength{tr_plot};    % 1 x nWin
nFrames = numel(trace);
nWin    = numel(ev);

% To match window to frame index, we map each window to its center frame
centerFrames = (1:nWin) + (winLen-1)/2;  % With winLen=10, shift by 4.5

figure;
subplot(2,1,1);
plot(1:nFrames, trace, '-');
xlabel('Frame index');
ylabel('Motion energy');
title(sprintf('Trial %d: Original frame-by-frame evidence', tr_plot));
grid on;

subplot(2,1,2);
plot(centerFrames, ev, '-o');
xlabel('Frame index (window center)');
ylabel('Sliding-window evidence');
title(sprintf('Trial %d: %d-frame sliding window (step = 1 frame)', tr_plot, winLen));
grid on;




%% 7. Build normalized-time residual evidence (for all trials)

% Read response and correct vectors
resp_vec    = allStruct.req_resp(:);   % 1 = right, 2 = left
correct_vec = allStruct.correct(:);    % 1 = correct, 0 = incorrect

% Number of points on normalized time axis
nBins  = 40;
t_norm = linspace(0, 1, nBins);        % 0 = trial start, 1 = trial end

% 7.1 Interpolate sliding-window evidence of all trials to 0–1 time axis
EV_all_norm = nan(nTrials, nBins);     % rows = trials, columns = time bins

for tr = 1:nTrials
    ev_tr = evidence_strength{tr};
    if isempty(ev_tr)
        continue;
    end
    nWin_tr = numel(ev_tr);
    t_orig  = linspace(0, 1, nWin_tr);
    
    EV_all_norm(tr, :) = interp1(t_orig, ev_tr, t_norm, 'linear');
end

% 7.2 Baseline across all trials at each time bin
baseline_t = mean(EV_all_norm, 1, 'omitnan');   % 1 x nBins

% 7.3 Residual evidence for each trial: evidence - baseline
EV_center = EV_all_norm - baseline_t;           % MATLAB auto-expands baseline_t

% Mask for trials that are not all NaN
mask_valid_center = ~all(isnan(EV_center), 2);


%% 8. Define 4 groups: Stim/Response (Right/Left) and compute kernels

% True stimulus side per trial: 1 = right, 2 = left
stim_vec = resp_vec;                  % start from response
wrong_idx = (correct_vec == 0);       % flip for incorrect trials
stim_vec(wrong_idx) = 3 - resp_vec(wrong_idx);   % 1 <-> 2

isStimR = (stim_vec == 1);
isStimL = (stim_vec == 2);
isRespR = (resp_vec == 1);
isRespL = (resp_vec == 2);

idx_SR = isStimR & isRespR & mask_valid_center;  % Stim R, Resp R
idx_SL = isStimL & isRespR & mask_valid_center;  % Stim L, Resp R
idx_RS = isStimR & isRespL & mask_valid_center;  % Stim R, Resp L
idx_LL = isStimL & isRespL & mask_valid_center;  % Stim L, Resp L

fprintf('\nTrials per group (Stim/Resp):\n');
fprintf('Stim R, Resp R: %d\n', nnz(idx_SR));
fprintf('Stim L, Resp R: %d\n', nnz(idx_SL));
fprintf('Stim R, Resp L: %d\n', nnz(idx_RS));
fprintf('Stim L, Resp L: %d\n', nnz(idx_LL));

% Mean residual evidence over time for each group
mean_SR = mean(EV_center(idx_SR, :), 1, 'omitnan');
mean_SL = mean(EV_center(idx_SL, :), 1, 'omitnan');
mean_RS = mean(EV_center(idx_RS, :), 1, 'omitnan');
mean_LL = mean(EV_center(idx_LL, :), 1, 'omitnan');


%% 9. Plot: one figure with 4 subplots (evidence kernels)

figure;

% 1) Stim R, Resp R
subplot(2,2,1);
plot(t_norm, mean_SR, '-b', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim R, Resp R');
grid on;

% 2) Stim L, Resp R
subplot(2,2,2);
plot(t_norm, mean_SL, '-r', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim L, Resp R');
grid on;

% 3) Stim R, Resp L
subplot(2,2,3);
plot(t_norm, mean_RS, '-r', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim R, Resp L');
grid on;

% 4) Stim L, Resp L
subplot(2,2,4);
plot(t_norm, mean_LL, '-b', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim L, Resp L');
grid on;


%% 9. Plot: one figure with 4 subplots (evidence kernels)

figure;

ax = gobjects(4,1);

% 1) Stim R, Resp R
ax(1) = subplot(2,2,1);
plot(t_norm, mean_SR, '-b', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim R, Resp R');
grid on;

% 2) Stim L, Resp R
ax(2) = subplot(2,2,2);
plot(t_norm, mean_SL, '-r', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim L, Resp R');
grid on;

% 3) Stim R, Resp L
ax(3) = subplot(2,2,3);
plot(t_norm, mean_RS, '-r', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim R, Resp L');
grid on;

% 4) Stim L, Resp L
ax(4) = subplot(2,2,4);
plot(t_norm, mean_LL, '-b', 'LineWidth', 1.2); hold on;
yline(0, '--k');
xlabel('Normalized time (0–1)');
ylabel('Residual evidence');
title('Stim L, Resp L');
grid on;

% normalize y axis
linkaxes(ax, 'y');
