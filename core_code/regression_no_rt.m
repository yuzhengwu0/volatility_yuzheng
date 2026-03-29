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
%   Linear regression predicting logit-transformed confidence:
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
% DEPENDENT VARIABLE:
%   Confidence in [0,1], slightly shrunk from boundaries and then
%   logit-transformed.
%
% KEY PREDICTORS:
%   perf : predicted performance
%   corr : correctness
%   vol  : residual volatility from motion energy

clear; clc; close all;

%% ===================== 0. Add toolboxes =====================
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/Palamedes1'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/boundedline-pkg-master'));
RPF_check_toolboxes;

%% ===================== SWITCH =====================
% plot
DO_PLOT_BIG_FIGURE = false;
DO_PLOT_AICBIC_DOTS  = true;
DO_PLOT_QUARTER_BAR = false;

useSubjDummies = false;

% predictors 
DO_SPLIT_COH = false; %coh
LOW_COH_VALUES = [0, 32, 64];
HIGH_COH_VALUES = [128, 256, 512];
DO_USE_RT = true; % RT
P_PERF_MODE = 'online'; % perf: 'all' or 'online' or 'try'
% parameters for resVol
nBins = 50;
winLen = 10;
tol    = 1e-12;

%% ===================== 1. Load data =====================
addpath('helper_functions/');
addpath('helper_functions/data_prep/')
addpath('helper_functions/fit_model/')
addpath('helper_functions/plot/')
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

valid_basic = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
    ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all);

valid_conf = (confCont_all >= 0) & (confCont_all <= 1);

% ===== try low coh here =====
if DO_SPLIT_COH
    valid_coh = ismember(coh_all, LOW_COH_VALUES); %change here if we want to change to high coh trials
else
    valid_coh = true(size(coh_all));
end

valid = valid_basic & valid_conf & valid_coh;

coh           = coh_all(valid);
resp          = resp_all(valid);
Correct       = correct_all(valid);
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = ME_cell_all(valid);
rt            = rt_all(valid);

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

%% ===================== 2. Cleaning Data =====================
% prep z_score confidence 
ConfY = transform_conf(confCont, subjID);

% prep z-score RT
if DO_USE_RT
    rtX = transform_rt(rt, subjID);
else
    rtX = [];
end

% prep z-scored residual volatility (resVol)
[resVol_mat, resVol, cond] = compute_resVol(motion_energy, vol, nBins, winLen, tol);

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
z_coh = zscore(coh_weuse);

% z-scored volatility condition 
z_cond = zscore(cond);

% SUMMARY!!!
% for later on anaysis we are using:
% ConfY (z-scored, n * 1)
% rtX (z-scored, n * 1)
% resVol (z-scored, n * nBins)
% p_perf_all/p_perf_online (z-scored, n * 1)
% coh
% Correct (raw accuracy (0/1), n * 1)

%% ===================== 2.5 Build & plot predictors =====================
% plot predictors & outcome variable
tiledlayout;
% plot raw confidence
nexttile;
histogram(confCont);
title('raw confidence (not used)')
% plot z-scored confidence
nexttile;
histogram(ConfY);
title('confidence z-scored within subject');
% plot other performance term
nexttile;
histogram(p_perf_all);
title('p perf all');
plot "online" performance term
nexttile; 
histogram(p_perf_online); 
title('p perf online'); 
% plot correctness
nexttile;
histogram(Correct);
title('correctness');
% plot volatility
nexttile;
histogram(resVol);
title('z scored volatility (resVol)')
% plot coherence
nexttile;
histogram(coh);
title('z scored volatility (resVol)')
% resVol check histogram
figure;
resVol_check;

%% ===================== 3. Define model family =====================
% run this to build model family and change the family model by un-comment

% model family 1:
% fixed terms: RT (R), accuracy (C), coherence (coh)
% M0: baseline + R + C + coh
% M1: baseline + R + C + coh + P
% M2: baseline + R + C + coh + V
% M3: baseline + R + C + coh + P + V
% M4: baseline + R + C + coh + P + V + P*V
% M5: baseline + R + C + P + V + P*V + P*V*coh
% [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
%     twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_coh();

% model family 2:
% fixed terms: RT (R), accuracy (C), coherence (coh), vol condition (z_cond)
% M0: baseline + R + C + coh + z_cond
% M1: baseline + R + C + coh + z_cond + P
% M2: baseline + R + C + coh + z_cond + V
% M3: baseline + R + C + coh + z_cond + P + V
% M4: baseline + R + C + coh + z_cond + P + V + P*V
% [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
%     twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_zcond();

% model family 3:
% fixed terms: RT (R), accuracy (C), coherence (coh)
% M0: baseline + R + C + coh
% M1: baseline + R + C + coh + P
% M2: baseline + R + C + coh + V
% M3: baseline + R + C + coh + P + V
% M4: baseline + R + C + coh + P + V + P*V
% M5: baseline + R + C + coh + P + V + C*V
[modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_corr();

% [modelNames, modelSpec, baseLabels, withcohNames, withcohLabels, oneWayNames, oneWayLabels, ...
%     twoWayNames, twoWayLabels, nocohNames, nocohLabels, threeWayNames, threeWayLabels] = build_model_family_no_rt_withCorrect();
nModels = numel(modelNames);

%% ===================== Prep cfg =====================
cfg = struct();

cfg.ConfY = ConfY;
cfg.Correct = Correct;
cfg.subjID = subjID;
cfg.z_coh = z_coh;
cfg.z_perf = z_perf;
cfg.rtX = rtX;
cfg.z_cond = z_cond;

cfg.resVol = resVol;
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

%% ===================== 4. Fit all models for AIC/BIC =====================
% run this to fit model in the model family
[Models, Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_corr(cfg);
cfg.Fitted_models = Fitted_models;

%% ===================== 5. Rank models by composite AIC/BIC score and dot plot =====================
% run this to see the winning AIC and BIC median and mean score for each model
[deltaTbl, score, rankIdx, top4Idx] = rank_models(AIC_mat, BIC_mat, cfg.modelNames);
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
QUARTER_MODEL_MODE = 'top1';      % 'top1' or 'manual'
QUARTER_MODEL_NAME = 'M2_V';   % only used if QUARTER_MODEL_MODE = 'manual'
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
