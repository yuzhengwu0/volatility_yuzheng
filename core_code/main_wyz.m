%% script using cohframes
clear; clc; close all;

addpath(genpath('helper_functions/'));
data_path = '../all_with_me.mat';

%% re-run this chunk if configuration changed
run('cfg_default.m');

% add valid trials filter
[cfg, idx] = prep_main(cfg, data_path);

% redefine accuracy
prep_answer_and_ME;

cond = nan(size(cfg.vol));
cond(cfg.vol == min(cfg.vol)) = 1;
cond(cfg.vol == max(cfg.vol)) = 2;
cfg.cond = cond;

% flipping left trials
if cfg.FLIPPING
    flip_leftward_trials;
end

%===== data prep

% conf (ConfY: z-scored conf)
[ConfY, confCont] = transform_conf(cfg);
cfg.ConfY = ConfY;
cfg.confCont = confCont;

% RT (rtX: z-loged rt)
rtX = transform_rt(cfg);
cfg.rtX = rtX;

% switch between moco_diff and ME!
% if use coh_diff
trial_coh = cfg.coh ./ 1000;
if VOL_USE_ME == false
    % coh
    coh_mat = repmat(trial_coh, [1, 17]);
    
    % vol1 & vol2
    % get momentary coherence - interpolate each trial to 17 frames (for RT task)
    moco_mat = NaN(nTrials, 17);
    for t = 1:height(cfg.cohframes)
        frames = cfg.cohframes{t};          % variable-length vector for this trial
        n = numel(frames);
        if n == 17
            moco_mat(t, :) = frames';
        else
            x_orig = linspace(1, 17, n);    % original timepoints scaled to 1–17
            x_new  = 1:17;                  % target timepoints
            moco_mat(t, :) = interp1(x_orig, frames, x_new, 'linear');
        end
    end
    
    % get direction of stim on each trial
    trial_dir = cfg.dir;
    trial_dir(trial_dir==180) = -1;
    trial_dir(trial_dir==0) = 1;
    % make trial coherence signed (-1 = left, +1 = right)
    coh_mat_dir = coh_mat.* trial_dir;
    % use the same signage for momentary coherence
    moco_mat_dir = moco_mat .* trial_dir;
    
    switch cfg.VOLMODE
        case 'vol1'
            % compute frame-by-frame difference in coherence
            moco_diff = moco_mat_dir - coh_mat_dir;
            vol = moco_diff;
        case 'vol2'
            vol2_signed = NaN(nTrials, 17);
            for t = 1:nTrials
                for f = 2:17
                    vol2_signed(t, f) = moco_mat_dir(t, f) - moco_mat_dir(t, f-1);
                end
            end
            vol2_abs = abs(vol2_signed);
            vol = vol2_signed; % or change to vol2_abs
        case 'vol3dir'
            vol3 = NaN(nTrials, 17);
            for t = 1:nTrials
                for i = 1:17
                    if i == 1
                        vol3(t,i) = moco_mat_dir(t, i);
                    else 
                        vol3(t,i) = moco_mat_dir(t, i) + vol3(t, i-1);
                    end
                end
            end
            vol = vol3;
        case 'vol3abs'
            vol3 = NaN(nTrials, 17);
            for t = 1:nTrials
                for i = 1:17
                    if i == 1
                        vol3(t,i) = moco_mat_dir(t, i);
                    else 
                        vol3(t,i) = moco_mat_dir(t, i) + vol3(t, i-1);
                    end
                end
            end
            vol = abs(vol3);
    end 

% if use ME
else
    thiscoh = unique(trial_coh);
    thiscond = unique(cfg.cond);
    
    motion_diff = nan(size(motion_mat));
    
    for icoh = 1:numel(thiscoh)
        for icond = 1:numel(thiscond)
            idx = (trial_coh == thiscoh(icoh)) & (cfg.cond == thiscond(icond));
            data = motion_mat(idx, :);
            cond_coh_mean_ME = mean(data, 1, 'omitnan');
            motion_diff(idx, :) = data - cond_coh_mean_ME;
            vol = motion_diff;
        end
    end 
end 




zvol = zscore(vol);
cfg.vol = zvol;
cfg.trial_coh = trial_coh;


%====== update correct
% exclude coh == 512
if cfg.DROP_HIGHEST_COH
    keep_coh = (cfg.coh ~= 0.512);
else
    keep_coh = true(size(cfg.coh));
end

Correct = cfg.Correct;
switch cfg.CORR
    case 'corr'
        keep_corr = (Correct == 1);
    case 'incorr'
        keep_corr = (Correct == 0);
    case 'all'
        keep_corr = true(size(Correct));
    otherwise
        error('Unknown cfg.CORR: %s', cfg.CORR);
end

switch cfg.MODEL_FAMILY
    case 'wyz'
        cfg_orig = cfg;
        coh_levels = [-1];  % -1 = all
    case 'wyzcoh'
        cfg_orig = cfg;
        % coh_levels = [256];  % -1 = all
        coh_levels = [0, 32, 64, 128, 256, 512, -1];
    case 'wyzcond'
        cfg_orig = cfg;
        coh_levels = [128];  % -1 = all
        % coh_levels = [0, 32, 64, 128, 256, 512, -1];
end

for ci = 1:numel(coh_levels)
    cfg = cfg_orig;
    if coh_levels(ci) == -1
        keep_coh = true(size(cfg.coh));
        coh_label = 'all';
        cfg.coh_label = coh_label;

    else
        keep_coh = (cfg.coh == coh_levels(ci));
        coh_label = num2str(coh_levels(ci));
        cfg.coh_label = coh_label;
    end


    keep = keep_coh & keep_corr;
    cfg.keep = keep;
    valid = cfg.keep;

    % update other parameters
    cfg.ConfY               = cfg.ConfY(valid);
    cfg.confCont            = cfg.confCont(valid);
    cfg.Correct             = Correct(valid);
    cfg.rtX                 = cfg.rtX(valid);
    cfg.subjID              = cfg.subjID(valid);
    cfg.coh                 = cfg.coh(valid);
    cfg.cond                = cfg.cond(valid);
    cfg.vol                 = cfg.vol(valid, :);
    cfg.req                 = cfg.req(valid);
    cfg.given               = cfg.given(valid);
    cfg.trial_coh           = cfg.trial_coh(valid);


    % output amount of trials
    nTrials = sum(cfg.keep);
    cfg.nTrials = nTrials;
    fprintf('Total valid trials: %d\n', nTrials);


    %===== model fitting
    switch cfg.OUTCOME

        % confidence model families
        case 'conf'

            switch cfg.MODEL_FAMILY
                case 'wyz'
                    [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
                        twoWayNames, twoWayLabels] = build_model_family_wyz();
                case 'wyzcoh'
                    [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels] = build_model_family_wyzcoh();
                case 'wyzcond'
                    [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, twoWayNames, twoWayLabels] = build_model_family_wyzcond();
            end
    end

    nModels = numel(modelNames);

    % ===================== Prep cfg =====================

    cfg.nModels = nModels;
    cfg.modelNames = modelNames;
    cfg.modelSpec = modelSpec;

    cfg.baseLabels = baseLabels;
    cfg.oneWayLabels = oneWayLabels;
    cfg.oneWayNames = oneWayNames;
    % cfg.twoWayNames = twoWayNames;
    % cfg.twoWayLabels = twoWayLabels;


    cfg.minN = 50;

    cfg.outPDF_ab = '../figure/AIC_BIC_bestModel_dots.pdf';


    switch cfg.OUTCOME
        case 'conf'
            switch cfg.MODEL_FAMILY
                case 'wyz'
                    [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_wyz(cfg);
                case 'wyzcoh'
                    [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_wyzcoh(cfg);
                case 'wyzcond'
                    [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_wyzcond(cfg);
            end
    end

    cfg.Fitted_models = Fitted_models;

    % ===================== 5. Rank models by composite AIC/BIC score and dot plot =====================
    % run this to see the winning AIC and BIC median and mean score for each model
    [deltaTbl, score, rankIdx, top4Idx, deltaBIC] = rank_models(AIC_mat, BIC_mat, cfg.modelNames);
    cfg.rankIdx = rankIdx;
    cfg.top4Idx = top4Idx;


    % run this to see the AIC BIC dot plot
    if cfg.DO_PLOT_AICBIC_DOTS
        plot_best_model_dots(AIC_mat, BIC_mat, cfg.modelNames, cfg.outPDF_ab, cfg);
    end
end 
