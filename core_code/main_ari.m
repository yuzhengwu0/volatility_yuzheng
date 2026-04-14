%% regression_no_rt_clean.m
% PURPOSE:
%   Analyze how time-resolved stimulus volatility influences continuous
%   confidence while accounting for predicted performance, correctness,
%   and response time.
%
% MAIN PROCEDURE:
%   1) Compute predicted performance using subject × volatility × coherence
%      mean accuracy (p_perf_all).
%   2) Extract time-resolved motion-energy statistics using a sliding
%      window and normalize each trial to 40 within-trial time bins.
%   3) Compute residual volatility (resVol_time) by regressing motion-energy
%      SD on absolute motion-energy mean at each time bin.
%
% REGRESSION ANALYSIS:
%   Linear regression predicting confidence [0-1]:
%
%       ConfY ~ perf + corr + vol + rt + interactions
%
%   Two levels of regression are performed:
%     (1) pooled regression across all subjects at each time bin
%         (used for AIC/BIC model comparison)
%     (2) per-subject regression at each time bin for the selected top models
%         (used for plotting beta time courses).
%
% OPTIONAL VISUALIZATIONS:
%   - AIC/BIC dot plot:
%       Shows which model has the minimum AIC (blue) and BIC (red)
%       at each time bin.
%
%   - Quarter-bin bar plot:
%       Pools bins into four within-trial quarters and shows the beta
%       of a selected predictor with SE and significance.
%
%   - Big figure:
%       Displays time courses of regression coefficients (per subject
%       and mean) for the top-ranked models.
%
% KEY PREDICTORS:
%   perf : predicted performance
%   corr : correctness
%   vol  : residual volatility from motion energy
%  coh  : stimulus coherence
%  z_cond: trial-level stimulus volatility

clear; clc; close all;

%% ===================== SWITCH =====================

% clear;
%% plot
DO_PLOT_BIG_FIGURE = false;
DO_PLOT_AICBIC_DOTS  = true;
DO_PLOT_QUARTER_BAR = false;
DO_PLOT_PREDICTORS = false;

useSubjDummies = true;

% predictors 
DO_SPLIT_COH = false; %coh
LOW_COH_VALUES = [0, 32, 64];
HIGH_COH_VALUES = [128, 256, 512];
DO_USE_RT = true; % RT
P_PERF_MODE = 'online'; % perf: 'all' or 'online' or 'try'
DISCRETE_COH = false; % do you want to discretize coherence into low/high?
% parameters for resVol
nBins = 27;
winLen = 5;
tol    = 1e-12;

% model family
MODEL_FAMILY = 'blend'; 
% outcome var
OUTCOME = 'conf'; % can be 'conf', 'acc', 'rt'

% ===================== 1. Load & clean data =====================
addpath('helper_functions/');
addpath('helper_functions/data_prep/')
addpath('helper_functions/fit_model/')
addpath('helper_functions/plot/')
data_path = '../all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh_all       = allStruct.rdm1_coh(:);
given_all      = allStruct.given_resp(:);        % 1/2
req_all       = allStruct.req_resp(:);
correct_all   = allStruct.correct(:);         % 1/0
confCont_all  = allStruct.confidence(:);      % 0-1
vol_all       = allStruct.rdm1_coh_std(:);
subjID_all    = allStruct.group(:);
motion_energy_all = allStruct.motion_energy;
rt_all        = allStruct.rt(:);

CORR = 'incorr'; % 'corr' or 'incorr' or 'all'
switch CORR
    case 'corr'
        valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ~isnan(confCont_all) & correct_all == 1 & ...
        ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;
    case 'incorr'
        valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ~isnan(confCont_all) & correct_all == 0 & ...
        ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;
    case 'all'
        valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ...
    ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;
end


% ===== try low coh here =====
if DO_SPLIT_COH
    valid_coh = ismember(coh_all, LOW_COH_VALUES); %change here if we want to change to high coh trials
else
    valid_coh = true(size(coh_all));
end

valid = valid_basic & valid_coh;

coh           = coh_all(valid);
req           = req_all(valid);
given         = given_all(valid);
Correct       = correct_all(valid);
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = motion_energy_all(valid);
rt            = rt_all(valid);

truesessiontrial = allStruct.trialnum(valid);
truesession = allStruct.session(valid);

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% ===================== 2. Prep predictors =====================
% prep z_score confidence 
ConfY = transform_conf(confCont, subjID);

% prep z-score RT
if DO_USE_RT
    rtX = transform_rt(rt, subjID);
else
    rtX = [];
    
end

% PREP VOL
% convert to cond (1=low vol, 2=high vol)
vol_levels = unique(vol(~isnan(vol)));
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;
%residual volatility & z-score
[resVol_mat, resVol, evidence_strength, volatility_strength] = compute_resVol(motion_energy, vol, nBins, winLen, tol, coh, cond, req);

% prep predicted performance
switch P_PERF_MODE
    case 'all'
        p_perf_all = compute_p_perf_all(subjID, cond, coh, Correct);
        z_perf = zscore(p_perf_all);
    case 'online'
        p_perf_online = compute_p_perf_online(subjID, cond, coh, Correct);
        z_perf = zscore(p_perf_online);
    case 'try' % if we want to see the distribution plot we can do all of them together, but usually do one at a time to avoid error
        p_perf_all = compute_p_perf_all(subjID, cond, coh, Correct);
        p_perf_online = compute_p_perf_online(subjID, cond, coh, Correct);
end

% accuracy --> we use correct directly

% coherence
coh_weuse = coh/100;
if DISCRETE_COH
    z_coh = coh_weuse >= median(coh_weuse);
else
    z_coh = zscore(coh_weuse);
end

% z-scored volatility condition 
z_cond = zscore(cond);

% z-log momentary volatility 
log_vol = log(resVol_mat + 1);
zlog_vol = zscore(log_vol);

% SUMMARY!!!
% for later on anaysis we are using:
% ConfY (z-scored, n * 1)
% rtX (z-scored, n * 1)
% resVol (z-scored, n * nBins)
% p_perf_all/p_perf_online (z-scored, n * 1)
% coh
% Correct (raw accuracy (0/1), n * 1)

% ===================== 2.5 Plot predictors =====================
if DO_PLOT_PREDICTORS
    plot_regression_variables
end

%% ===================== 3. Define model families =====================
% run this to build model family and change the family model by un-comment
% outer loop controls outcome variable
switch OUTCOME 

    % confidence model families
    case 'conf'

        switch MODEL_FAMILY

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
cfg = struct();

cfg.ConfY = ConfY;
cfg.Correct = Correct;
cfg.subjID = subjID;
cfg.z_coh = z_coh;
cfg.z_perf = z_perf;
cfg.rtX = rtX;
cfg.z_cond = z_cond;

cfg.resVol = zlog_vol;
cfg.nModels = nModels;
cfg.modelNames = modelNames;
cfg.modelSpec = modelSpec;

cfg.baseLabels = baseLabels;
cfg.oneWayLabels = oneWayLabels;
cfg.oneWayNames = oneWayNames;
cfg.twoWayNames = twoWayNames;
cfg.threeWayLabels = threeWayLabels;
cfg.threeWayNames = threeWayNames;

cfg.useSubjDummies = useSubjDummies;
cfg.minN = 50;

cfg.DO_PLOT_AICBIC_DOTS = DO_PLOT_AICBIC_DOTS;
cfg.outPDF_ab = '../figure/AIC_BIC_bestModel_dots.pdf';

% ===================== 4. Fit all models for AIC/BIC =====================
% run this to fit model in the model family
switch OUTCOME
    case 'acc'
        [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_accuracy(cfg);
    case 'rt'
        [Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_rt(cfg);
    case 'conf'
        switch MODEL_FAMILY
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

%% ===================== 6. Quarter-bin pooled effect bar plot =====================
% run this chunk to get a quarter divided plot showing polled effect of
% selected term

% switch
QUARTER_MODEL_MODE = 'manual';      % 'top1' or 'manual'
QUARTER_MODEL_NAME = 'M4_twoWay_PxV';   % only used if QUARTER_MODEL_MODE = 'manual'
QUARTER_TERM_NAME  = 'PxV';       % e.g. 'V','R','C','coh','PxV','VxC','RxV','PxVxC'...
cfg.QUARTER_MODEL_MODE = QUARTER_MODEL_MODE;
cfg.QUARTER_MODEL_NAME = QUARTER_MODEL_NAME;
cfg.QUARTER_TERM_NAME = QUARTER_TERM_NAME;

% plot
if DO_PLOT_QUARTER_BAR
    [quarterTbl, figQ, betaBins_q] = plot_quarter_bar(cfg);
end

%% ===================== 7. Refit top 4 per subject per bin =====================
% refit
cfg.top4Idx  = top4Idx;
cfg.minN_sub = 5;
cfg.sv_tol   = 1e-12;

Sel = refit_top_models_by_subj_coh(cfg, Models);

%% ===================== 8. One big figure =====================
% plot big figure
if DO_PLOT_BIG_FIGURE
    t_norm = linspace(0, 1, size(cfg.resVol, 2));

    colSub = [ ...
        0.0000    0.4470    0.7410
        1.0000    0.0000    0.0000
        0.9290    0.6940    0.1250
        ];

    desiredNames = modelNames(top4Idx);

    SelOrdered = Sel([]);
    for i = 1:numel(desiredNames)
        hit = find(strcmp({Sel.mName}, desiredNames{i}), 1, 'first');
        if isempty(hit)
            error('Sel missing model %s. Did Section 9 run?', desiredNames{i});
        end
        SelOrdered(i) = Sel(hit);
    end

    outPDF   = '../figure/BigFigure_Top4_CurrentFamily_AllTerms.pdf';
    figTitle = 'Top 4 models - current coh family';

    % all terms/ change to some of terms
    termList = [];

    plot_bigfigure_top4_allTerms_coh(SelOrdered, t_norm, colSub, outPDF, termList, figTitle);
end


%% ====== one model all betas, with error bar and AIC BIC ======
get_fitted_params;


%% two plot (model) vertically aligned, all terms in each model
% use it to compare AIC BIC and winning terms within model family
models_to_plot = [2, 5];
upper_lower_all;


%% two plot (model) vertically aligned, only one choosen term and all three subjects
% check what terms in each model (check this first, the model number is differ)
for i = 1:numel(Sel)
    fprintf('\n=== Sel(%d): %s ===\n', i, Sel(i).mName);
    for j = 1:numel(Sel(i).termLabels)
        fprintf('  [%d] "%s"\n', j, Sel(i).termLabels{j});
    end
end

colSub = lines(3);  % 3 subjects 3 colors
plot_models_oneTerm(Sel, [1, 2], {'b_{vol}', 'b_{perf\times vol}'}, t_norm, colSub, AIC_mat, BIC_mat);
