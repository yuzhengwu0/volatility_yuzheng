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

%% ===================== PLOT SWITCH =====================
DO_PLOT_BIG_FIGURE = false;
DO_PLOT_AICBIC_DOTS  = true;

useSubjDummies = false;

% predictors 
DO_SPLIT_COH = false; %coh
LOW_COH_VALUES = [0, 32, 64];
HIGH_COH_VALUES = [128, 256, 512];
DO_USE_RT = true; % RT
P_PERF_MODE = 'all'; % perf: 'all' or 'online' or 'try'
% parameters for resVol
nBins = 50;
winLen = 10;
tol    = 1e-12;

DO_PLOT_QUARTER_BAR = false;
QUARTER_MODEL_MODE = 'manual';      % 'top1' or 'manual'
QUARTER_MODEL_NAME = 'M7_all2';   % only used if QUARTER_MODEL_MODE = 'manual'
QUARTER_TERM_NAME  = 'vol';       % e.g. 'vol','rt','PxV','VxC','RxV','PxVxC'

%% ===================== 1. Load data =====================
addpath('helper_functions/');
addpath('helper_functions/data_prep/')
addpath('helper_functions/fit_model/')
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
% plot "online" performance term
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
% resVol check histogram
figure;
resVol_check;

%% ===================== 3. Define model family =====================
% run this to build model family
[modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_coh();

% [modelNames, modelSpec, baseLabels, withcohNames, withcohLabels, oneWayNames, oneWayLabels, ...
%     twoWayNames, twoWayLabels, nocohNames, nocohLabels, threeWayNames, threeWayLabels] = build_model_family_no_rt_withCorrect();
nModels = numel(modelNames);

%% ===================== Prep cfg =====================
cfg = struct();

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

cfg.ConfY = ConfY;
cfg.Correct = Correct;
cfg.subjID = subjID;
cfg.coh = coh;
cfg.z_coh = z_coh;
cfg.z_perf = z_perf;
cfg.rtX = rtX;

cfg.useSubjDummies = useSubjDummies;
cfg.minN = 50;

cfg.DO_PLOT_AICBIC_DOTS = DO_PLOT_AICBIC_DOTS;
cfg.outPDF_ab = '../figure/AIC_BIC_bestModel_dots.pdf';

%% ===================== 4. Fit all models for AIC/BIC =====================
% run this to fit model in the model family
[Models, Fitted_models, AIC_mat, BIC_mat, Nobs_mat] = fit_model_coh(cfg);

%% ===================== 5. Rank models by composite AIC/BIC score and dot plot =====================
% run this to see the winning AIC and BIC score for each model
[deltaTbl, score, rankIdx, top4Idx] = rank_models(AIC_mat, BIC_mat, cfg.modelNames);

% run this to see the AIC BIC dot plot
if cfg.DO_PLOT_AICBIC_DOTS
    plot_best_model_dots(AIC_mat, BIC_mat, cfg.modelNames, cfg.outPDF_ab);
end

%% ===================== 8.5 Quarter-bin pooled effect bar plot =====================
if DO_PLOT_QUARTER_BAR

    % ---------- choose model ----------
    switch QUARTER_MODEL_MODE
        case 'top1'
            qModelIdx = rankIdx(1);
        case 'manual'
            qModelIdx = find(strcmp(modelNames, QUARTER_MODEL_NAME), 1, 'first');
            if isempty(qModelIdx)
                error('QUARTER_MODEL_NAME not found: %s', QUARTER_MODEL_NAME);
            end
        otherwise
            error('Unknown QUARTER_MODEL_MODE: %s', QUARTER_MODEL_MODE);
    end

    qModelName = modelNames{qModelIdx};
    fprintf('\n=== Quarter-bar regression using model: %s ===\n', qModelName);

    % ---------- define 4 quarter bins ----------
    qEdges = {1:10, 11:20, 21:30, 31:40};
    qLabels = {'Q1','Q2','Q3','Q4'};

    beta_q = nan(1,4);
    se_q   = nan(1,4);
    p_q    = nan(1,4);
    n_q    = nan(1,4);

    for q = 1:4
        bins_here = qEdges{q};

        % average volatility within this quarter
        Vq_raw = mean(resVol_time(:, bins_here), 2, 'omitnan');

        mask = ~isnan(Vq_raw) & ~isnan(ConfY) & ~isnan(Correct) & ...
            ~isnan(p_perf_online) & ~isnan(subjID) & ~isnan(rtX);

        if sum(mask) < minN
            fprintf('Quarter %s skipped: not enough trials.\n', qLabels{q});
            continue;
        end

        y    = ConfY(mask);
        % P    = Fp_all(mask);
        P = p_perf_online(mask);
        % C    = Cz_all(mask);
        C = Correct(mask);
        Vraw = Vq_raw(mask);
        % R    = RTz_all(mask);
        R = rtX(mask);
        sID  = subjID(mask);

        sv = std(Vraw);
        if sv < 1e-12
            fprintf('Quarter %s skipped: V variance too small.\n', qLabels{q});
            continue;
        end
        V = (Vraw - mean(Vraw)) ./ sv;

        % interactions
        PxC = P.*C; PxV = P.*V; PxR = P.*R;
        VxC = V.*C; CxR = C.*R; RxV = R.*V;

        PxVxC   = P.*V.*C;
        PxCxR   = P.*C.*R;
        PxVxR   = P.*V.*R;
        VxCxR   = V.*C.*R;
        PxVxCxR = P.*V.*C.*R;

        if useSubjDummies
            S2 = double(sID==2);
            S3 = double(sID==3);

            T = table(y,P,C,V,R,PxC,PxV,PxR,VxC,CxR,RxV,PxVxC,PxCxR,PxVxR,VxCxR,PxVxCxR,S2,S3, ...
                'VariableNames', {'ConfY','perf','corr','vol','rt', ...
                'PxC','PxV','PxR','VxC','CxR','RxV', ...
                'PxVxC','PxCxR','PxVxR','VxCxR','PxVxCxR','S2','S3'});
        else
            T = table(y,P,C,V,R,PxC,PxV,PxR,VxC,CxR,RxV,PxVxC,PxCxR,PxVxR,VxCxR,PxVxCxR, ...
                'VariableNames', {'ConfY','perf','corr','vol','rt', ...
                'PxC','PxV','PxR','VxC','CxR','RxV', ...
                'PxVxC','PxCxR','PxVxR','VxCxR','PxVxCxR'});
        end

        % same formula logic as AIC/BIC section
        f = "ConfY ~ perf + corr + vol + rt";
        for j = 1:6
            if modelSpec(qModelIdx).use2(j), f = f + " + " + twoWayNames(j); end
        end
        for j = 1:4
            if modelSpec(qModelIdx).use3(j), f = f + " + " + threeWayNames(j); end
        end
        if modelSpec(qModelIdx).use4, f = f + " + " + fourWayNames; end
        if useSubjDummies, f = f + " + S2 + S3"; end

        g = fitglm(T, f, 'Distribution','normal');

        coefNames = string(g.CoefficientNames);
        hit = find(coefNames == QUARTER_TERM_NAME, 1, 'first');

        if isempty(hit)
            error('Term "%s" is not in model %s.', QUARTER_TERM_NAME, qModelName);
        end

        beta_q(q) = g.Coefficients.Estimate(hit);
        se_q(q)   = g.Coefficients.SE(hit);
        p_q(q)    = g.Coefficients.pValue(hit);
        n_q(q)    = sum(mask);
    end

    % ---------- plot ----------
    figQ = figure('Color','w','Position',[200 200 720 480]); hold on;

    bh = bar(1:4, beta_q, 0.65, 'FaceColor', [0.65 0.65 0.65], 'EdgeColor', 'none');
    errorbar(1:4, beta_q, se_q, 'k.', 'LineWidth', 1.2, 'CapSize', 12);

    yline(0, 'k--', 'LineWidth', 1);

    set(gca, 'XTick', 1:4, 'XTickLabel', qLabels, 'FontSize', 12, 'LineWidth', 1);
    xlabel('Within-trial quarter');
    ylabel(sprintf('Beta: %s', QUARTER_TERM_NAME), 'Interpreter', 'none');
    title(sprintf('%s | %s', qModelName, QUARTER_TERM_NAME), 'Interpreter', 'none');

    % significance stars
    yTop = max(beta_q + se_q, [], 'omitnan');
    yBot = min(beta_q - se_q, [], 'omitnan');
    yrng = yTop - yBot;
    if ~isfinite(yrng) || yrng <= 0
        yrng = 1;
    end

    for q = 1:4
        if ~isnan(p_q(q)) && p_q(q) < 0.05
            text(q, beta_q(q) + se_q(q) + 0.06*yrng, '*', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 18, ...
                'FontWeight', 'bold');
        end
    end

    ylim([yBot - 0.12*yrng, yTop + 0.18*yrng]);
    box off;

    outPDF_ab = '../figure/AIC_BIC_bestModel_dots.pdf';
    outPDF_q = '../figure/QuarterBar_%s_%s.pdf';
    set(figQ,'Renderer','painters');
    print(figQ, outPDF_q, '-dpdf', '-painters');
    fprintf('✓ Saved quarter bar plot: %s\n', outPDF_q);

    % print result table
    quarterTbl = table((1:4)', qLabels(:), beta_q(:), se_q(:), p_q(:), n_q(:), ...
        'VariableNames', {'QuarterIdx','Quarter','Beta','SE','pValue','N'});
    disp(quarterTbl);
end

%% ===================== 9. Refit top 4 per subject per bin =====================
modelIdxToRefit = top4Idx(:)';

subj_list = unique(subjID(:))';
nSubj     = numel(subj_list);
K         = numel(t_norm);

colSub   = [0 0.4470 0.7410; 1 0 0; 0.9290 0.6940 0.1250];
minN_sub = 5;
sv_tol   = 1e-12;

Sel = struct();

for ii = 1:numel(modelIdxToRefit)
    mIdx  = modelIdxToRefit(ii);
    mName = modelNames{mIdx};

    termLabels = baseLabels;
    termNames  = ["(Intercept)","perf","corr","vol","rt"];

    for j = 1:6
        if modelSpec(mIdx).use2(j)
            termLabels{end+1} = twoWayLabels{j};
            termNames(end+1)  = twoWayNames(j);
        end
    end

    for j = 1:4
        if modelSpec(mIdx).use3(j)
            termLabels{end+1} = threeWayLabels{j};
            termNames(end+1)  = threeWayNames(j);
        end
    end

    if modelSpec(mIdx).use4
        termLabels{end+1} = fourWayLabels{1};
        termNames(end+1)  = fourWayNames;
    end

    nTerms   = numel(termNames);
    beta_sub = nan(nSubj, K, nTerms);
    se_sub   = nan(nSubj, K, nTerms);
    p_sub    = nan(nSubj, K, nTerms);

    fprintf('\n--- Refit %s per subject/per bin ---\n', mName);

    for iSub = 1:nSubj
        thisSub = subj_list(iSub);

        for k = 1:K
            Vk = resVol_time(:,k);

            mask = ~isnan(Vk) & ~isnan(ConfY) & ~isnan(Correct) & ...
                ~isnan(Fp_all) & ~isnan(subjID) & ~isnan(rt);
            mask = mask & (subjID == thisSub);

            if sum(mask) < minN_sub, continue; end

            y    = ConfY(mask);
            % P    = Fp_all(mask);
            P = p_perf_online(mask);
            % C    = Cz_all(mask);
            C = Correct(mask);
            Vraw = Vk(mask);

            sv = std(Vraw);
            if sv < sv_tol, continue; end
            V = (Vraw - mean(Vraw)) ./ sv;

            RTraw = rt(mask);
            RTref = log(RTraw + 1e-6);
            R     = (RTref - mean(RTref,'omitnan')) ./ std(RTref,'omitnan');

            PxC = P.*C; PxV = P.*V; PxR = P.*R;
            VxC = V.*C; CxR = C.*R; RxV = R.*V;

            PxVxC   = P.*V.*C;
            PxCxR   = P.*C.*R;
            PxVxR   = P.*V.*R;
            VxCxR   = V.*C.*R;
            PxVxCxR = P.*V.*C.*R;

            T = table(y,P,C,V,R,PxC,PxV,PxR,VxC,CxR,RxV,PxVxC,PxCxR,PxVxR,VxCxR,PxVxCxR, ...
                'VariableNames', {'ConfY','perf','corr','vol','rt', ...
                'PxC','PxV','PxR','VxC','CxR','RxV', ...
                'PxVxC','PxCxR','PxVxR','VxCxR','PxVxCxR'});

            f = "ConfY ~ perf + corr + vol + rt";
            for j = 1:6
                if modelSpec(mIdx).use2(j), f = f + " + " + twoWayNames(j); end
            end
            for j = 1:4
                if modelSpec(mIdx).use3(j), f = f + " + " + threeWayNames(j); end
            end
            if modelSpec(mIdx).use4, f = f + " + " + fourWayNames; end

            try
                g = fitglm(T, f, 'Distribution','normal');
            catch
                continue;
            end

            coefNames = string(g.CoefficientNames);
            coefEst   = g.Coefficients.Estimate;
            coefSE    = g.Coefficients.SE;
            coefP     = g.Coefficients.pValue;

            for tt = 1:nTerms
                nm = termNames(tt);
                hit = find(coefNames == nm, 1, 'first');
                if ~isempty(hit)
                    beta_sub(iSub,k,tt) = coefEst(hit);
                    se_sub(iSub,k,tt)   = coefSE(hit);
                    p_sub(iSub,k,tt)    = coefP(hit);
                end
            end
        end
    end

    Sel(ii).mName      = mName;
    Sel(ii).termLabels = termLabels;
    Sel(ii).beta_sub   = beta_sub;
    Sel(ii).se_sub     = se_sub;
    Sel(ii).p_sub      = p_sub;
    Sel(ii).beta_pool = Models(m).betas;      % K x nTerms
    Sel(ii).se_pool   = Models(m).beta_ses;   % K x nTerms
end

%% ===================== 10. One big figure =====================
if DO_PLOT_BIG_FIGURE
    desiredNames = modelNames(top4Idx);

    SelOrdered = Sel([]);
    for i = 1:numel(desiredNames)
        hit = find(strcmp({Sel.mName}, desiredNames{i}), 1, 'first');
        if isempty(hit)
            error('Sel missing model %s. Did Section 9 run?', desiredNames{i});
        end
        SelOrdered(i) = Sel(hit);
    end

    outPDF   = 'BigFigure_Top4_VolTermsOnly.pdf';
    figTitle = 'Top 4 models - volatility-related terms only';

    termListVol = { ...
        'b_{vol}', ...
        'b_{perf×vol}', ...
        'b_{vol×corr}', ...
        'b_{rt×vol}', ...
        'b_{perf×vol×corr}', ...
        'b_{perf×vol×rt}', ...
        'b_{vol×corr×rt}', ...
        'b_{perf×vol×corr×rt}' ...
        };

    plot_bigfigure_top4_allTerms(SelOrdered, t_norm, colSub, outPDF, termListVol, figTitle);
end

%% ===================== LOCAL FUNCTIONS =====================

function plot_bigfigure_top4_allTerms(Sel, t_norm, colSub, outPDF, termList, figTitle)

if nargin < 4 || isempty(outPDF)
    outPDF = 'BigFigure_Top4_AllTerms.pdf';
end
if nargin < 5
    termList = [];
end
if nargin < 6 || isempty(figTitle)
    figTitle = 'Top 4 models - all terms';
end

nCols = numel(Sel);
if nCols == 0
    warning('Sel is empty. Nothing to plot.');
    return;
end

nSubj = size(Sel(1).beta_sub,1);
x     = t_norm(:)';

% ---------- collect terms ----------
allTerms = {};
for c = 1:nCols
    allTerms = [allTerms, Sel(c).termLabels]; %#ok<AGROW>
end
allTerms = unique(allTerms, 'stable');

if ~isempty(termList)
    keep = ismember(allTerms, termList);
    allTerms = allTerms(keep);
end

keep2 = false(size(allTerms));
for r = 1:numel(allTerms)
    term = allTerms{r};
    hasAny = false;
    for c = 1:nCols
        if any(strcmp(Sel(c).termLabels, term))
            hasAny = true;
            break;
        end
    end
    keep2(r) = hasAny;
end
allTerms = allTerms(keep2);

nRows = numel(allTerms);
if nRows == 0
    warning('No terms to plot.');
    return;
end

% ---------- row-specific y limits ----------
yLimByRow = cell(nRows,1);
for r = 1:nRows
    yLimByRow{r} = local_term_ylim_allModels(Sel, allTerms{r});
end

% ---------- layout ----------
tileSize = 130;
gapX     = 20;
gapY     = 14;
labelW   = 90;

outerL = 40;
outerR = 30;
outerT = 35;
outerB = 100;

figW = outerL + labelW + nCols*tileSize + (nCols-1)*gapX + outerR;
figH = outerT + nRows*tileSize + (nRows-1)*gapY + outerB;

fig = figure('Color','w','Units','points','Position',[60 60 figW figH]);

pt2nx = @(pt) pt / figW;
pt2ny = @(pt) pt / figH;

L = pt2nx(outerL);
R = pt2nx(outerR);
T = pt2ny(outerT);
B = pt2ny(outerB);

gapXNorm   = pt2nx(gapX);
gapYNorm   = pt2ny(gapY);
labelWNorm = pt2nx(labelW);
tileWNorm  = pt2nx(tileSize);
tileHNorm  = pt2ny(tileSize);

usedW = labelWNorm + nCols*tileWNorm + (nCols-1)*gapXNorm;
usedH = nRows*tileHNorm + (nRows-1)*gapYNorm;

x0   = L + ((1 - L - R) - usedW)/2;
yTop = 1 - T - ((1 - T - B) - usedH)/2;

fontPanel = 9;
lw_sub    = 0.55;
lw_mean   = 1.10;
alphaSub  = 0.12;
alphaMean = 0.08;

lastAxPerCol = gobjects(1,nCols);
legendPlaced = false;

for r = 1:nRows
    yPos = yTop - r*tileHNorm - (r-1)*gapYNorm;

    % row label
    axLab = axes('Parent',fig,'Units','normalized','Position',[x0, yPos, labelWNorm, tileHNorm]);
    axis(axLab,'off');
    text(axLab, 0.50, 0.50, term_to_tex_compact(allTerms{r}), ...
        'FontSize', 16, ...
        'Rotation', 90, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'Interpreter','latex', ...
        'Clipping','on');

    for c = 1:nCols
        xPos = x0 + labelWNorm + (c-1)*(tileWNorm + gapXNorm);

        ax = axes('Parent',fig,'Units','normalized','Position',[xPos, yPos, tileWNorm, tileHNorm]);
        hold(ax,'on'); grid(ax,'on'); box(ax,'off');

        if r == 1
            ttl = strrep(Sel(c).mName, '_', '\_');
            title(ax, ['$' ttl '$'], ...
                'Interpreter','latex', ...
                'FontSize',10, ...
                'FontWeight','bold');
        end

        tt = find(strcmp(Sel(c).termLabels, allTerms{r}), 1, 'first');

        if isempty(tt)
            axis(ax,'off');

            if ~legendPlaced
                hold(ax,'on');

                hLeg = gobjects(nSubj+1,1);
                legText = cell(nSubj+1,1);

                for s = 1:nSubj
                    hLeg(s) = plot(ax, nan, nan, '-', 'LineWidth', 2.5, 'Color', colSub(s,:));
                    legText{s} = sprintf('$\\mathrm{Subject\\ %d}$', s);
                end
                hLeg(nSubj+1) = plot(ax, nan, nan, 'k-', 'LineWidth', 3.0);
                legText{nSubj+1} = '$\mathrm{Mean}$';

                legend(ax, hLeg, legText, ...
                    'Box','off', ...
                    'FontSize',10, ...
                    'Location','northwest', ...
                    'Interpreter','latex');

                legendPlaced = true;
            end

            continue;
        end

        xlim(ax,[0 1]);
        xticks(ax,0:0.2:1);
        set(ax,'FontSize',fontPanel,'LineWidth',0.8);
        set(ax,'TickLabelInterpreter','latex');
        yline(ax,0,'k--','LineWidth',0.6,'HandleVisibility','off');

        if r < nRows
            set(ax,'XTickLabel',[]);
        end
        lastAxPerCol(c) = ax;


        beta_sub = Sel(c).beta_sub(:,:,tt);
        se_sub   = Sel(c).se_sub(:,:,tt);

        % subjects
        for s = 1:nSubj
            yv = squeeze(beta_sub(s,:));
            ev = squeeze(se_sub(s,:));

            ok = ~isnan(yv) & ~isnan(ev);
            if sum(ok) >= 2
                xx = x(ok); yy = yv(ok); ee = ev(ok);
                fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(s,:), ...
                    'EdgeColor','none', 'FaceAlpha',alphaSub, 'HandleVisibility','off');
            end

            plot(ax, x, yv, '-', ...
                'Color', colSub(s,:), ...
                'LineWidth', lw_sub, ...
                'HandleVisibility','off');
        end

        % mean ± SEM (original ver. average of three subjects
        % and SE here is the difference between three subjects)
        % yMean = mean(beta_sub,1,'omitnan');
        % ySEM  = std(beta_sub,0,1,'omitnan') ./ sqrt(nSubj);
        %
        % okm = ~isnan(yMean) & ~isnan(ySEM);
        % if sum(okm) >= 2
        %     xx = x(okm); ym = yMean(okm); es = ySEM(okm);
        %     fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
        %         'EdgeColor','none', 'FaceAlpha',alphaMean, 'HandleVisibility','off');
        % end
        % plot(ax, x, yMean, 'k-', 'LineWidth', lw_mean, 'HandleVisibility','off');

        % pooled line ± pooled SE
        yPool = Sel(c).beta_pool(:,tt)';
        ePool = Sel(c).se_pool(:,tt)';

        okm = ~isnan(yPool) & ~isnan(ePool);
        if sum(okm) >= 2
            xx = x(okm); ym = yPool(okm); es = ePool(okm);
            fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                'EdgeColor','none', 'FaceAlpha',alphaMean, 'HandleVisibility','off');
        end
        plot(ax, x, yPool, 'k-', 'LineWidth', lw_mean, 'HandleVisibility','off');

        ylim(ax, yLimByRow{r});
    end
end

for c = 1:nCols
    ax = lastAxPerCol(c);
    if ~isempty(ax) && isgraphics(ax) && strcmp(get(ax,'Visible'),'on')
        xlabel(ax,'Normalized time (0--1)', 'Interpreter','latex', 'FontSize',10);
    end
end

if exist('sgtitle','file') == 2
    sgtitle(figTitle, 'FontWeight','bold', 'Interpreter','latex');
else
    annotation(fig,'textbox',[0 0.97 1 0.03], ...
        'String', figTitle, ...
        'EdgeColor','none', ...
        'HorizontalAlignment','center', ...
        'FontWeight','bold');
end

set(fig,'Renderer','painters');
set(fig,'PaperUnits','points');
set(fig,'PaperSize',[figW figH]);
set(fig,'PaperPosition',[0 0 figW figH]);
set(fig,'PaperPositionMode','manual');
set(fig,'PaperOrientation','portrait');

print(fig, outPDF, '-dpdf', '-painters');
fprintf('✓ Saved: %s\n', outPDF);

end

function yLim = local_term_ylim_allModels(Sel, termLabel)

vals = [];

for c = 1:numel(Sel)
    tt = find(strcmp(Sel(c).termLabels, termLabel), 1, 'first');
    if isempty(tt), continue; end

    b = Sel(c).beta_sub(:,:,tt);
    e = Sel(c).se_sub(:,:,tt);

    vals = [vals; b(:); b(:)+e(:); b(:)-e(:)]; %#ok<AGROW>
end

vals = vals(~isnan(vals));
if isempty(vals)
    yLim = [-1 1];
    return;
end

lo = prctile(vals, 2);
hi = prctile(vals, 98);

if abs(hi-lo) < 1e-6
    lo = lo - 1;
    hi = hi + 1;
end

pad = 0.10 * (hi - lo + eps);
yLim = [lo-pad, hi+pad];

end

function out = term_to_tex_compact(lbl)

switch lbl
    case 'b0 (Intercept)'
        out = '$b_0$';

    case 'b_{perf}'
        out = '$b_{\mathrm{perf}}$';

    case 'b_{corr}'
        out = '$b_{\mathrm{corr}}$';

    case 'b_{vol}'
        out = '$b_{\mathrm{vol}}$';

    case 'b_{rt}'
        out = '$b_{\mathrm{rt}}$';

    case 'b_{perf×corr}'
        out = '$b_{\mathrm{perf}\times\mathrm{corr}}$';

    case 'b_{perf×vol}'
        out = '$b_{\mathrm{perf}\times\mathrm{vol}}$';

    case 'b_{perf×rt}'
        out = '$b_{\mathrm{perf}\times\mathrm{rt}}$';

    case 'b_{vol×corr}'
        out = '$b_{\mathrm{vol}\times\mathrm{corr}}$';

    case 'b_{corr×rt}'
        out = '$b_{\mathrm{corr}\times\mathrm{rt}}$';

    case 'b_{rt×vol}'
        out = '$b_{\mathrm{rt}\times\mathrm{vol}}$';

    case 'b_{perf×vol×corr}'
        out = '$b_{\mathrm{perf}\times\mathrm{vol}\times\mathrm{corr}}$';

    case 'b_{perf×corr×rt}'
        out = '$b_{\mathrm{perf}\times\mathrm{corr}\times\mathrm{rt}}$';

    case 'b_{perf×vol×rt}'
        out = '$b_{\mathrm{perf}\times\mathrm{vol}\times\mathrm{rt}}$';

    case 'b_{vol×corr×rt}'
        out = '$b_{\mathrm{vol}\times\mathrm{corr}\times\mathrm{rt}}$';

    case 'b_{perf×vol×corr×rt}'
        out = '$b_{\mathrm{perf}\times\mathrm{vol}\times\mathrm{corr}\times\mathrm{rt}}$';

    otherwise
        out = ['$' lbl '$'];
end

end