clear; clc; close all;

addpath(genpath('helper_functions/'));
data_path = '../all_with_me.mat';
%% re-run this chunk if configuration changed
run('cfg_default.m');

% add valid trials filter
cfg = prep_main(cfg, data_path);

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

%% data prep

% vol convert to cond (cond: 1=low vol, 2=high vol)
vol_levels = unique(cfg.vol(~isnan(cfg.vol)));
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end


% conf (ConfY: z-scored conf)
[ConfY, confCont] = transform_conf(cfg);
cfg.ConfY = ConfY;
cfg.confCont = confCont;


% RT (rtX: z-loged rt)
rtX = transform_rt(cfg);
cfg.rtX = rtX;


% do resVol
[resVol_mat, resVol, evidence_strength, volatility_strength, cfg] = ...
    compute_resVol(cfg);
cfg.resVol_mat          = resVol_mat;
cfg.resVol              = resVol;
cfg.evidence_strength   = evidence_strength;
cfg.volatility_strength = volatility_strength;


% predicted performance
switch cfg.P_PERF_MODE
    case 'all'
        p_perf_all = compute_p_perf_all(cfg);
        z_perf = zscore(p_perf_all);
    case 'online'
        [p_perf_online, combination_counter, combination_performance] = compute_p_perf_online(cfg);
        z_perf = zscore(p_perf_online);
end
cfg.z_perf = z_perf;


% z-log momentary volatility 
log_vol = log(cfg.resVol_mat + 1);
cfg.zlog_vol = zscore(log_vol);


%% parameters check figs

% plot predictors and outcome variables distribution check
plot_regression_variables

% mean STD resVol check (red blue big graph)
mean_STD_resVol_plot

% volatility distribution check (scatter plot)
vol_distribution_scatter_plot

%% exclude criteria
% exclude coh == 512
if cfg.DROP_HIGHEST_COH
    keep_coh = (cfg.coh_weuse ~= 5.12);
else
    keep_coh = true(size(cfg.coh));
end


% update correct
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
keep = keep_coh & keep_corr;
cfg.keep = keep;
valid = cfg.keep;

% update other parameters
cfg.ConfY               = cfg.ConfY(valid);
cfg.confCont            = cfg.confCont(valid);
cfg.Correct             = Correct(valid);
cfg.z_perf              = cfg.z_perf(valid);
cfg.rtX                 = cfg.rtX(valid);
cfg.subjID              = cfg.subjID(valid);
cfg.coh                 = cfg.coh(valid);
cfg.cond                = cfg.cond(valid);
cfg.vol                 = cfg.vol(valid);
cfg.req                 = cfg.req(valid);
cfg.given               = cfg.given(valid);
cfg.coh_weuse           = cfg.coh_weuse(valid);
cfg.resVol              = cfg.resVol(valid, :);
cfg.resVol_mat          = cfg.resVol_mat(valid, :);
cfg.zlog_vol            = cfg.zlog_vol(valid, :);
cfg.evidence_strength   = cfg.evidence_strength(valid, :);
cfg.volatility_strength = cfg.volatility_strength(valid, :);


% output amount of trials
nTrials = sum(cfg.keep);
fprintf('Total valid trials: %d\n', nTrials);


%% parameters check figs

% plot predictors and outcome variables distribution check
plot_regression_variables

% check raw motion energy mean & STD distribution, linear or polynomial
ME_mean_STD_regression

% resVol check
resVol_check

% mean STD resVol check (red blue big graph)
mean_STD_resVol_plot



%% POLLED -- group
% bar graph (mean accuracy)
plot_group_mean_accuracy

% bar graph (mean confidence)
plot_group_means_byCond

% purple and green regression plot
get_individual_betas


%% fitting and running models
% run this to build model family and change the family model by un-comment
% outer loop controls outcome variable
switch cfg.OUTCOME 

    % confidence model families
    case 'conf'

        switch cfg.MODEL_FAMILY

            case 'cond' 
                % model family 1: perf & vol
                % fixed terms: RT (R), accuracy (C), coherence (coh), vol condition (z_cond)
                % M0: baseline + R + C + coh + z_cond
                % M1: baseline + R + C + coh + z_cond + P
                % M2: baseline + R + C + coh + z_cond + V
                % M3: baseline + R + C + coh + z_cond + P + V
                % M4: baseline + R + C + coh + z_cond + P + V + P*V
                % M5: baseline + R + C + coh + z_cond + P + V + C*V
                % M6: baseline + R + C + coh + z_cond + P + V + P*V*coh
                % M7: baseline + R + C + coh + z_cond + P + V + C*V*coh
                [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
                     twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_zcond();

            case 'coh'
                % model family 2: volatility only
                % fixed terms: RT (R), accuracy (C), coherence (coh), vol condition (z_cond)
                % M0: baseline + R + C + coh + z_cond
                % M1: baseline + R + C + coh + z_cond + V
                % M2: baseline + R + C + coh + z_cond + V + C*V
                % M3: baseline + R + C + coh + z_cond + V + C*V + C*V*coh
                [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
                    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_coh();

            case 'blend'
                [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
                    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_blend_corr();
        end

     % accuracy model
    case 'acc'
        [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
            twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_accuracy();
    % rt model
    case 'rt'
        [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
        twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_rt();
end

nModels = numel(modelNames);

% ===================== Prep cfg =====================

cfg.nModels = nModels;
cfg.modelNames = modelNames;
cfg.modelSpec = modelSpec;

cfg.baseLabels = baseLabels;
cfg.oneWayLabels = oneWayLabels;
cfg.oneWayNames = oneWayNames;
cfg.twoWayNames = twoWayNames;
cfg.threeWayLabels = threeWayLabels;
cfg.threeWayNames = threeWayNames;

cfg.minN = 50;

cfg.outPDF_ab = '../figure/AIC_BIC_bestModel_dots.pdf';

% ===================== 4. Fit all models for AIC/BIC =====================
% run this to fit model in the model family
switch cfg.OUTCOME
    case 'acc'
        [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_accuracy(cfg);
    case 'rt'
        [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_rt(cfg);
    case 'conf'
        switch cfg.MODEL_FAMILY
            case 'cond'
                [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_zcond(cfg);
            case 'coh'
                [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_coh(cfg);
            case 'blend'
                [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_blend_corr(cfg);
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
    plot_best_model_dots(AIC_mat, BIC_mat, cfg.modelNames, cfg.outPDF_ab);
end

%% plots

%% checking parameters