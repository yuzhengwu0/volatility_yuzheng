%% resVol_coh0_confCorr_vol.m
% Goal:
% 1) From all.motion_energy, compute sliding-window mean & std
% 2) Warp to normalized time (0–1, 40 bins)
% 3) At each time bin, run regression: STD ~ |MEAN| + coherence → residual volatility
% 4) Only keep trials with coh = 0
% 5) Split trials into 8 groups: high/low volatility × high/low confidence × correct/incorrect
% 6) Plot 8 time courses in one figure:
%    Red = high volatility, Blue = low volatility
%    Each line is one conf × correct group, with a clear legend label

clear; clc;

%% 0. Load data -----------------------------------------------------------
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';   % TODO: change to your own path
tmp       = load(data_path, 'all');
allStruct = tmp.all;

% TODO: Change field names below to match your own struct
motion_energy = allStruct.motion_energy;    % nTrials x 1 cell, each: nFrames x 1 double
cohVec        = allStruct.rdm1_coh(:);      % coherence per trial
volVec        = allStruct.rdm1_coh_std(:);  % volatility condition (two levels: low/high)
confRating    = allStruct.confidence(:);    % continuous confidence (0–1)
correctVec    = allStruct.correct(:);       % 0/1: 1=correct, 0=incorrect

nTrials = numel(motion_energy);
fprintf('Loaded %d trials.\n', nTrials);

%% 1. Sliding window: for each trial, compute mean & std, remove padding zeros at the end -----
winLen    = 10;               % each window has 10 frames
mean_cell = cell(nTrials, 1);
std_cell  = cell(nTrials, 1);

for tr = 1:nTrials
    ev = motion_energy{tr};    % nFrames x 1 double (with padding zeros at the end)
    
    if isempty(ev)
        mean_cell{tr} = [];
        std_cell{tr}  = [];
        continue;
    end
    
    % ====== New: remove trailing padding zeros ======
    % Find the last non-zero position
    lastNonZero = find(ev ~= 0, 1, 'last');
    
    if isempty(lastNonZero)
        % If all values are 0, treat as empty trial
        mean_cell{tr} = [];
        std_cell{tr}  = [];
        continue;
    else
        % Keep only up to the last non-zero frame
        ev = ev(1:lastNonZero);
    end
    % ====== End of new part =================================
    
    nFrames = numel(ev);
    nWin    = nFrames - winLen + 1;
    if nWin <= 0
        mean_cell{tr} = [];
        std_cell{tr}  = [];
        continue;
    end
    
    mVec = nan(1, nWin);
    sVec = nan(1, nWin);
    for w = 1:nWin
        seg     = ev(w : w+winLen-1);
        mVec(w) = mean(seg);
        sVec(w) = std(seg);
    end
    
    mean_cell{tr} = mVec;
    std_cell{tr}  = sVec;
end

%% 2. Warp to normalized time (0–1, 40 bins) ------------------------
nBins  = 40;
mean40 = nan(nTrials, nBins);   % trial x time
std40  = nan(nTrials, nBins);

for tr = 1:nTrials
    mVec = mean_cell{tr};
    sVec = std_cell{tr};
    
    if isempty(mVec) || isempty(sVec)
        continue;
    end
    
    x_old = linspace(0, 1, numel(mVec));
    x_new = linspace(0, 1, nBins);
    
    mean40(tr, :) = interp1(x_old, mVec, x_new, 'linear', 'extrap');
    std40(tr, :)  = interp1(x_old, sVec, x_new, 'linear', 'extrap');
end

timeAxis = linspace(0, 1, nBins);   % normalized time

%% 3. At each time bin, run STD ~ |MEAN| + coherence to get residual volatility ---
resVol = nan(nTrials, nBins);   % trial x time

for t = 1:nBins
    y      = std40(:, t);           % STD at time t
    X_mean = abs(mean40(:, t));     % |MEAN| at time t
    X_coh  = cohVec;                % coherence per trial
    
    valid = ~isnan(y) & ~isnan(X_mean) & ~isnan(X_coh);
    if nnz(valid) < 10
        continue;
    end
    
    X  = [ones(nnz(valid),1), X_mean(valid), X_coh(valid)];  % [intercept, |MEAN|, coh]
    yy = y(valid);
    
    % OLS regression
    b     = X \ yy;
    yhat  = X * b;
    resid = yy - yhat;              % residual volatility = extra noise
    
    resVol(valid, t) = resid;
end

%% 4. Keep only trials with coh = 0 --------------------------------------
idx_coh0 = (cohVec == 0);
fprintf('Trials with coh = 0: %d\n', nnz(idx_coh0));

% Use median split for confidence: ≥ median = high conf, < median = low conf
validConf  = ~isnan(confRating);
medConf    = median(confRating(validConf), 'omitnan');

isHighConf = confRating >= medConf;   % 1 = high conf, 0 = low conf
isCorrect  = (correctVec == 1);

% Volatility two levels: low / high (use min and max)
volLevels = unique(volVec(~isnan(volVec)));
volLevels = sort(volLevels(:)');
lowVolLevel  = volLevels(1);
highVolLevel = volLevels(end);

isLowVol  = (volVec == lowVolLevel);
isHighVol = (volVec == highVolLevel);

%% 5. Define trial indices for the 8 groups ----------------------------------------
% 1. HighVol, HighConf, Correct
idx1 = idx_coh0 & isHighVol & isHighConf & isCorrect;

% 2. HighVol, LowConf, Correct
idx2 = idx_coh0 & isHighVol & ~isHighConf & isCorrect;

% 3. HighVol, HighConf, Incorrect
idx3 = idx_coh0 & isHighVol & isHighConf & ~isCorrect;

% 4. HighVol, LowConf, Incorrect
idx4 = idx_coh0 & isHighVol & ~isHighConf & ~isCorrect;

% 5. LowVol, HighConf, Correct
idx5 = idx_coh0 & isLowVol & isHighConf & isCorrect;

% 6. LowVol, LowConf, Correct
idx6 = idx_coh0 & isLowVol & ~isHighConf & isCorrect;

% 7. LowVol, HighConf, Incorrect
idx7 = idx_coh0 & isLowVol & isHighConf & ~isCorrect;

% 8. LowVol, LowConf, Incorrect
idx8 = idx_coh0 & isLowVol & ~isHighConf & ~isCorrect;

% Print trial count for each group
groupCounts = [nnz(idx1), nnz(idx2), nnz(idx3), nnz(idx4), ...
               nnz(idx5), nnz(idx6), nnz(idx7), nnz(idx8)];
fprintf('Group trial counts (1–8): '); fprintf('%d ', groupCounts); fprintf('\n');

%% 6. For each group and each time bin, compute mean residual volatility --------------
mean1 = mean(resVol(idx1, :), 1, 'omitnan');
mean2 = mean(resVol(idx2, :), 1, 'omitnan');
mean3 = mean(resVol(idx3, :), 1, 'omitnan');
mean4 = mean(resVol(idx4, :), 1, 'omitnan');

mean5 = mean(resVol(idx5, :), 1, 'omitnan');
mean6 = mean(resVol(idx6, :), 1, 'omitnan');
mean7 = mean(resVol(idx7, :), 1, 'omitnan');
mean8 = mean(resVol(idx8, :), 1, 'omitnan');

%% 7. Plot: 8 lines in one figure ---------------------------------------------
figure; hold on; box on;

% --- Colors to better separate the lines ---
% High vol uses warm colors (4 lines, clearly different):
col_high = [ ...
    0.90 0.10 0.10;  % 1 red (HighConf-Correct)
    1.00 0.55 0.00;  % 2 bright orange (LowConf-Correct)
    0.80 0.00 0.40;  % 3 magenta (HighConf-Incorrect)
    0.60 0.30 0.00]; % 4 brown (LowConf-Incorrect)

% Low vol uses cool colors (4 lines, clearly different):
col_low = [ ...
    0.00 0.20 0.80;  % 5 dark blue (HighConf-Correct)
    0.00 0.60 0.50;  % 6 teal (LowConf-Correct)
    0.30 0.00 0.80;  % 7 purple (HighConf-Incorrect)
    0.00 0.60 0.90]; % 8 cyan (LowConf-Incorrect)

lw = 1.8;  % LineWidth

% High vol = warm colors, four groups: 1–4
h1 = plot(timeAxis, mean1, '-', 'Color', col_high(1,:), 'LineWidth', lw);  % 1. HighVol, HighConf, Correct
h2 = plot(timeAxis, mean2, '-', 'Color', col_high(2,:), 'LineWidth', lw);  % 2. HighVol, LowConf,  Correct
h3 = plot(timeAxis, mean3, '-', 'Color', col_high(3,:), 'LineWidth', lw);  % 3. HighVol, HighConf, Incorrect
h4 = plot(timeAxis, mean4, '-', 'Color', col_high(4,:), 'LineWidth', lw);  % 4. HighVol, LowConf,  Incorrect

% Low vol = cool colors, four groups: 5–8
h5 = plot(timeAxis, mean5, '-', 'Color', col_low(1,:), 'LineWidth', lw);   % 5. LowVol, HighConf, Correct
h6 = plot(timeAxis, mean6, '-', 'Color', col_low(2,:), 'LineWidth', lw);   % 6. LowVol, LowConf,  Correct
h7 = plot(timeAxis, mean7, '-', 'Color', col_low(3,:), 'LineWidth', lw);   % 7. LowVol, HighConf, Incorrect
h8 = plot(timeAxis, mean8, '-', 'Color', col_low(4,:), 'LineWidth', lw);   % 8. LowVol, LowConf,  Incorrect

xlabel('Normalized time');
ylabel('Residual volatility (extra noise)');
title('Residual volatility over time (coh = 0)');

legendStr = { ...
    'High vol - High conf - Correct', ...
    'High vol - Low conf  - Correct', ...
    'High vol - High conf - Incorrect', ...
    'High vol - Low conf  - Incorrect', ...
    'Low  vol - High conf - Correct', ...
    'Low  vol - Low conf  - Correct', ...
    'Low  vol - High conf - Incorrect', ...
    'Low  vol - Low conf  - Incorrect'};

legend([h1 h2 h3 h4 h5 h6 h7 h8], legendStr, 'Location', 'eastoutside');
set(gca, 'FontSize', 12);
