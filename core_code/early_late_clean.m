%% early_late_clean.m
% PURPOSE:
%   Compare EARLY vs LATE trials in the confidence-volatility analysis
%   using a switchable split mode:
%       (1) by raw trial order per subject
%       (2) by complete cycles of coherence x volatility combinations
%
% USER REQUESTED SETTINGS:
%   - confidence: within-subject z-score (NO logit)
%   - performance: subject x volatility x coherence mean accuracy (NO RPF)
%   - residual volatility: computed separately for EARLY and LATE
%   - final figure: EARLY M8 | LATE M8 | EARLY M9 | LATE M9
%   - plotting/y-axis: follow code2 logic, i.e. same y-range within each term
%     for EARLY vs LATE inside the same model
%
% REGRESSION:
%   confY ~ perf + corr + vol + rt + interactions
%
% NOTES:
%   - pooled stage uses AIC/BIC but fixedTopIdx = [9 10] by default (M8, M9)
%   - per-subject refits do not use subject dummies
%   - volatility residualization is split-specific (EARLY/LATE separately)

clear; clc; close all;

%% ===================== 0) Toolboxes: only run once at startup of matlab ======================
% addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/boundedline-pkg-master'));
addpath("helper_functions/");

%% ===================== USER SETTINGS: run every time you clear the workspace =====================

% ---------- Split mode ----------
% 'trial' : first nEarly / last nLate trials per subject
% 'cycle' : first nEarlyCycles / last nLateCycles complete cycles per subject
SPLIT_MODE = 'percent';   % <-- 'trial' or 'cycle' or 'percent'
DO_PLOT_AICBIC_DOTS = false;
DO_PRINT_SUMMARY = true;

% ---------- Trial-based split ----------
nEarlyTrials = 300;     % first n trials per subject
nLateTrials  = 500;     % last  m trials per subject

% ---------- Cycle-based split ----------
nEarlyCycles = 10;   % first n complete cycles
nLateCycles  = 10;   % last  m complete cycles

% ---------- Percentage-based split ----------
pEarly = 0.2;
pLate = 0.3;


% One cycle = all coherence x volatility combinations
cohLevels_cycle = [0 32 64 128 256 512];
volLevels_cycle = [0 256];

% ---------- Time bins for regression ----------
nBins        = 50;
minN_pooled  = 50;
minN_sub     = 5;

useSubjDummies = true;   % pooled model-selection stage only

% ---------- Fixed models ----------
FORCE_FIXED_MODELS = true;
fixedTopIdx = [9 10];    % M8, M9

% ---------- Plot ----------
DO_PLOT = false;
outPDF  = 'BigFigure_M8M9_EarlyLate_AllTerms.pdf';

% subject colors (3 subjects)
colSub = [0 0.4470 0.7410;
          1 0 0;
          0.9290 0.6940 0.1250];


%% ===================== 1) Load data ======================
data_path = '../all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh_all       = allStruct.rdm1_coh(:);
resp_all      = allStruct.req_resp(:);        % 1/2
correct_all   = allStruct.correct(:);         % 1/0
confCont_all  = allStruct.confidence(:);      % 0-1
vol_all       = allStruct.rdm1_coh_std(:);
subjID_all    = allStruct.group(:);
ME_cell_all   = allStruct.motion_energy;
rt_all        = allStruct.rt(:);

% ---- validity ----
valid_basic = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
              ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all);

valid_conf = (confCont_all >= 0) & (confCont_all <= 1);

valid = valid_basic & valid_conf;

fprintf('Dropped by conf out-of-range: %d trials (%.2f%% of basic-valid)\n', ...
    sum(valid_basic & ~valid_conf), ...
    100 * sum(valid_basic & ~valid_conf) / max(1, sum(valid_basic)));

coh           = coh_all(valid);
resp          = resp_all(valid);
Correct       = correct_all(valid);
% Cz_all        = Correct;  We use Correct itself instead of Cz_all
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = ME_cell_all(valid);
rt            = rt_all(valid);

nTrials = numel(Correct);
fprintf('Total valid trials: %d\n', nTrials);

subj_list = unique(subjID);
nSubj     = numel(subj_list);

%% ===================== 2) Confidence and RT: within-subject z-score ==============
ConfY   = nan(size(confCont));
RTz_all = nan(size(rt));

for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = subjID == s;

    % ---- confidence: within-subject z-score ----
    y = confCont(idxS);
    mu = mean(y, 'omitnan');
    sigma = std(y, 'omitnan');

    if sigma == 0 || isnan(sigma)
        ConfY(idxS) = zeros(size(y));
    else
        ConfY(idxS) = (y - mu) ./ sigma;
    end

    % ---- RT: log then within-subject z-score ----
    rt_sub = rt(idxS);
    rt_log = log(rt_sub);

    mu_rt = mean(rt_log, 'omitnan');
    sd_rt = std(rt_log, 'omitnan');

    if sd_rt == 0 || isnan(sd_rt)
        RTz_all(idxS) = zeros(size(rt_log));
    else
        RTz_all(idxS) = (rt_log - mu_rt) ./ sd_rt;
    end
end


%% ===================== 3) Volatility condition index ======================
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end

cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;


%% ===================== 4) Early late trial index and boundary========================
early_boundary_global = zeros(nSubj, 1);
late_boundary_global  = zeros(nSubj, 1);

for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = find(subjID == s);
    nS = numel(idxS);
    
    if strcmp(SPLIT_MODE, 'trial')
        nE = min(nEarlyTrials, nS);
        nL = min(nLateTrials,  nS);
        early_boundary_global(iSub) = idxS(nE);
        late_boundary_global(iSub)  = idxS(end - nL + 1);  % late started here
        
    elseif strcmp(SPLIT_MODE, 'cycle')
        coh_s = coh(idxS);
        vol_s = vol(idxS);
        
        % --- nEarlyCycles from begining ---
        seen = false(numel(cohLevels_cycle), numel(volLevels_cycle));
        cycleCount = 0;
        lastEarlyPos = nS;
        for p = 1:nS
            ic = find(coh_s(p) == cohLevels_cycle, 1);
            iv = find(vol_s(p) == volLevels_cycle, 1);
            if isempty(ic) || isempty(iv), continue; end
            seen(ic, iv) = true;
            if all(seen(:))
                cycleCount = cycleCount + 1;
                seen = false(numel(cohLevels_cycle), numel(volLevels_cycle));
                if cycleCount == nEarlyCycles
                    lastEarlyPos = p;
                    break;
                end
            end
        end
        early_boundary_global(iSub) = idxS(lastEarlyPos);
        
        % --- nLateCycles from the end count backward ---
        seen = false(numel(cohLevels_cycle), numel(volLevels_cycle));
        cycleCount = 0;
        firstLatePos = 1;
        for p = nS:-1:1  % from the end
            ic = find(coh_s(p) == cohLevels_cycle, 1);
            iv = find(vol_s(p) == volLevels_cycle, 1);
            if isempty(ic) || isempty(iv), continue; end
            seen(ic, iv) = true;
            if all(seen(:))
                cycleCount = cycleCount + 1;
                seen = false(numel(cohLevels_cycle), numel(volLevels_cycle));
                if cycleCount == nLateCycles
                    firstLatePos = p;
                    break;
                end
            end
        end
        late_boundary_global(iSub) = idxS(firstLatePos);  % late started here
        
    elseif strcmp(SPLIT_MODE, 'percent')
        nE = floor(pEarly * nS);
        nL = floor(pLate  * nS);
        early_boundary_global(iSub) = idxS(max(nE, 1));
        late_boundary_global(iSub)  = idxS(end - nL + 1);
    end
end

% build idxEarly / idxLate logical vectors
idxEarly = false(nTrials, 1);
idxLate  = false(nTrials, 1);

for iSub = 1:nSubj
    s = subj_list(iSub);
    idxS = find(subjID == s);
    idxEarly(idxS(idxS <= early_boundary_global(iSub))) = true;
    idxLate(idxS(idxS >= late_boundary_global(iSub)))   = true;
end

%% ===================== 4) Performance: condition mean accuracy ============
% compute accuracy as a function of trial in the decision task %%%%%%%%%%%

% initialize variables to store performance estimates
p_perf_online = zeros(nTrials, 1);
p_perf_online(:) = 0.5;

early_perf = nan(nTrials, 1); 
% NOTE: early and late is set as "TRIALS" right now,
% if we want to use the cycle thing, we have to change this!!!

% initialize counters for keeping track of trials: 12 combinations total
% column1: cond_list == cond_list(1) & coh_list == coh_list(1)
% column2: cond_list == cond_list(1) & coh_list == coh_list(2)
% .... through coh_list == coh_list(6)
% column7: cond_list == cond_list(2) & coh_list==coh_list(1)
% and so on...
cond_list = unique(cond(~isnan(cond)));
coh_list  = unique(coh(~isnan(coh)));
total_combinations = length(cond_list) * length(coh_list);

combination_counter = zeros(nSubj, total_combinations);
combination_performance = combination_counter;
combination_counter(:) = 2;      % 2 pseudo-trials to anchor at chance
combination_performance(:) = 1;  % 1 correct out of those 2 → 0.5
endTrial = zeros(nSubj, 1);

% initialize early cycle counter and early cycle perf
early_perf_online     = nan(nTrials, 1);
early_counter = nan(nSubj, length(cond_list) * length(coh_list)); 
early_perf_at_cycle   = nan(nSubj, length(cond_list) * length(coh_list));  

% compute "online" performance estimation
for iSub = 1:nSubj
    thisSub = subj_list(iSub);
    mask = (subjID == thisSub);
    nTrials_sub = sum(mask);

    if iSub > 1
        endTrial(iSub) = endTrial(iSub-1) + nTrials_sub;
        startTrial = endTrial(iSub-1) + 1;
    else
        endTrial(iSub) = nTrials_sub;
        startTrial = 1;
    end

    for tr = startTrial:endTrial(iSub)

        % (1) get cond, coh, correct values on this trial
        this_cond = cond(tr);
        this_coh  = coh(tr);
        this_correct = Correct(tr);

        % (2) find the index of this trial's cond and coh in their respective lists
        cond_idx = find(cond_list == this_cond, 1);
        coh_idx  = find(coh_list  == this_coh,  1);

        % compute the column index into combination_counter:
        % cond block of 6 + position within that block
        combo_idx = (cond_idx - 1) * length(coh_list) + coh_idx;

        % (3) increment the counter for this subject & combination
        combination_counter(iSub, combo_idx) = combination_counter(iSub, combo_idx) + 1;
        
        % (4) update performance estimate for this combination
        combination_performance(iSub, combo_idx) = combination_performance(iSub, combo_idx) + this_correct;
        p_perf_online(tr) = combination_performance(iSub, combo_idx) / combination_counter(iSub, combo_idx);
        
        % (4.5) extract early counter and early perf for cycle
        % save early_perf_at_cycle counter（without pseudo）
        if ~isnan(early_boundary_global(iSub)) && tr == early_boundary_global(iSub)
            early_perf_at_cycle(iSub, :) = combination_performance(iSub, :) ./ combination_counter(iSub, :);
            early_counter(iSub, :) = combination_counter(iSub, :) - 2;  % get rid of pseudo
        end

        % (5) store early trials directly
        if tr <= early_boundary_global(iSub)
            early_perf(tr) = p_perf_online(tr);
        end

    end
end

% save late_perf
late_perf_all = nan(nSubj, total_combinations);

for iSub = 1:nSubj
    late_perf_all(iSub, :) = combination_performance(iSub, :) ./ combination_counter(iSub, :);
end

%% ===================== 6) residual volatility: compute for all trials ======
winLen = 10;
tol    = 1e-12;
t_norm = linspace(0, 1, nBins);
resVol_mat = compute_resVol_time(motion_energy, nBins, winLen, tol);

%resVol_time_early = compute_resVol_time_split(motion_energy, subjID, nBins, winLen, tol, idxEarly);
%resVol_time_late  = compute_resVol_time_split(motion_energy, subjID, nBins, winLen, tol, idxLate);

%fprintf('Residual volatility (EARLY): %d trials x %d bins\n', size(resVol_time_early,1), size(resVol_time_early,2));
%fprintf('Residual volatility (LATE) : %d trials x %d bins\n', size(resVol_time_late,1), size(resVol_time_late,2));

%% ===================== PRINT SUMMARY =====================
if DO_PRINT_SUMMARY
    fprintf('\n========== DATA SUMMARY FOR REGRESSION ==========\n');
    fprintf('SPLIT_MODE: %s\n', SPLIT_MODE);
    
    for iSub = 1:nSubj
        s = subj_list(iSub);
        fprintf('\n--- Subject %d ---\n', s);
        
        % 1) Trial counts
        idxS_all   = find(subjID == s);
        idxS_early = find(subjID == s & (1:nTrials)' <= early_boundary_global(iSub));
        idxS_late  = find(subjID == s & idxLate);
        fprintf('  Trials total : %d\n', numel(idxS_all));
        fprintf('  Trials early : %d  (global boundary index = %d)\n', numel(idxS_early), early_boundary_global(iSub));
        fprintf('  Trials late  : %d\n', numel(idxS_late));
        
        % 2) Confidence (ConfY)
        conf_early = ConfY(idxS_early);
        conf_late  = ConfY(idxS_late);
        fprintf('  ConfY early  : %d values, mean=%.3f, std=%.3f  [stored in ConfY, nTrials x 1]\n', ...
            numel(conf_early), mean(conf_early,'omitnan'), std(conf_early,'omitnan'));
        fprintf('  ConfY late   : %d values, mean=%.3f, std=%.3f\n', ...
            numel(conf_late), mean(conf_late,'omitnan'), std(conf_late,'omitnan'));
        
        % 3) Residual volatility (resVol_mat)
        resVol_early = resVol_mat(idxS_early, :);
        resVol_late  = resVol_mat(idxS_late,  :);
        fprintf('  resVol early : %d trials x %d bins, mean=%.6f, std=%.6f  [stored in resVol_mat]\n', ...
            size(resVol_early,1), size(resVol_early,2), mean(resVol_early(:),'omitnan'), std(resVol_early(:),'omitnan'));
        fprintf('  resVol late  : %d trials x %d bins, mean=%.6f, std=%.6f\n', ...
            size(resVol_late,1), size(resVol_late,2), mean(resVol_late(:),'omitnan'), std(resVol_late(:),'omitnan'));
        
        % 4) Performance
        % early: cumulative up to boundary → early_perf_at_cycle (1 x 12 per subject)
        % late:  cumulative over all trials → late_perf_all (1 x 12 per subject)
        fprintf('  Perf early   : 1x12 combo vector  [stored in early_perf_at_cycle(%d,:)]\n', iSub);
        fprintf('                 %s\n', mat2str(round(early_perf_at_cycle(iSub,:), 3)));
        fprintf('  Perf late    : 1x12 combo vector  [stored in late_perf_all(%d,:)]\n', iSub);
        fprintf('                 %s\n', mat2str(round(late_perf_all(iSub,:), 3)));
        
        % 5) RT (RTz_all)
        rt_early = RTz_all(idxS_early);
        rt_late  = RTz_all(idxS_late);
        fprintf('  RT early     : %d values, mean=%.3f  [stored in RTz_all, nTrials x 1]\n', ...
            numel(rt_early), mean(rt_early,'omitnan'));
        fprintf('  RT late      : %d values, mean=%.3f\n', ...
            numel(rt_late), mean(rt_late,'omitnan'));
    end
    
    fprintf('\n==================================================\n');
end

%% ===================== 8) Model family ===================================
[modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, ...
    threeWayNames, threeWayLabels, fourWayNames, fourWayLabels] = build_model_family();

%% ===================== 9) Run pipeline for EARLY and LATE ================
if DO_PLOT

    resVol_time_early = resVol_mat(idxEarly);
    SelEarly = run_split_and_plot( ...
        idxEarly, 'EARLY', ...
        ConfY, Correct, subjID, early_perf, Cz_all, RTz_all, resVol_time_early, t_norm, colSub, ...
        modelNames, modelSpec, baseLabels, ...
        twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
        useSubjDummies, minN_pooled, minN_sub, ...
        FORCE_FIXED_MODELS, fixedTopIdx, DO_PLOT_AICBIC_DOTS, cfg);

    resVol_time_late = resVol_mat(idxLate);
    SelLate = run_split_and_plot( ...
        idxLate, 'LATE', ...
        ConfY, Correct, subjID, p_perf_online, Cz_all, RTz_all, resVol_time_late, t_norm, colSub, ...
        modelNames, modelSpec, baseLabels, ...
        twoWayNames, twoWayLabels, threeWayNames, threeWayLabels, fourWayNames, fourWayLabels, ...
        useSubjDummies, minN_pooled, minN_sub, ...
        FORCE_FIXED_MODELS, fixedTopIdx, DO_PLOT_AICBIC_DOTS, cfg);

    % Final figure: Early M8 | Late M8 | Early M9 | Late M9
    plot_bigfigure_4cols_M8M9_earlyLate(SelEarly, SelLate, t_norm, colSub, outPDF, [], ...
        sprintf('EARLY vs LATE | M8 vs M9 | split = %s', upper(SPLIT_MODE)));
end

fprintf('\nDone.\n');


%% ===================== NEW CODE WRITTEN TOGETHER 03/24 ===================== 

%% new approach

% initialize configuration structure that holds all the shared inputs
% the names don't have to be identical on both ends, but it is easier if they are
cfg.subjID              = subjID;
cfg.ConfY               = ConfY;
cfg.Correct             = Correct;
cfg.RTz_all             = RTz_all;
cfg.p_perf_online    = p_perf_online;
cfg.t_norm              = t_norm;
cfg.resVol_mat        = resVol_mat;
cfg.colSub              = colSub;
cfg.modelNames          = modelNames;
cfg.modelSpec           = modelSpec;
cfg.baseLabels          = baseLabels;
cfg.twoWayNames         = twoWayNames;
cfg.twoWayLabels        = twoWayLabels;
cfg.threeWayNames       = threeWayNames;
cfg.threeWayLabels      = threeWayLabels;
cfg.fourWayNames        = fourWayNames;
cfg.fourWayLabels       = fourWayLabels;
cfg.useSubjDummies      = useSubjDummies;
cfg.minN_pooled         = minN_pooled;
cfg.minN_sub            = minN_sub;
cfg.FORCE_FIXED_MODELS  = FORCE_FIXED_MODELS;
cfg.fixedTopIdx         = fixedTopIdx;
cfg.DO_PLOT_AICBIC_DOTS = DO_PLOT_AICBIC_DOTS;

%% fit pooled models
[~, ~, ~, early_fits] = fit_models_pooled(idxEarly, cfg);
[~, ~, ~, late_fits] = fit_models_pooled(idxLate, cfg);

%% run_split_and_plot for early & late
SelEarly = run_split_and_plot(idxEarly, 'EARLY', early_perf,    resVol_time_early, cfg);
SelLate  = run_split_and_plot(idxLate,  'LATE',  p_perf_online, resVol_time_late,  cfg);
