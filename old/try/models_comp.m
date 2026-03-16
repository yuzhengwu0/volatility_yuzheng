%% Fitting models comperation (FULL SCRIPT)
% Goals:
% 1) Use RPF (per subject) to generate predicted accuracy p_perf(j) for each trial
% 2) Use motion_energy to compute time-resolved residual volatility resVol_time (trial x time)
% 3) Run a logistic regression at each time bin:
%       Conf ~ f(p_j) + Correct + V + V*Correct (+ interactions by model)
%    Estimate time-resolved betas
%    Plot ONLY the final big multi-panel figure (PDF)

clear; clc;

%% PLOT SWITCHES (ONLY BIG FIGURE)
DO_PLOT_SECTION7       = false;  % per-model small figures
DO_PLOT_SECTION9_PASS2 = false;  % per-term individual figures
DO_PLOT_BIG_FIGURE     = true;   % final big PDF only

%% 0. Add toolboxes ------------------------------------------------------
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/Palamedes1'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/boundedline-pkg-master'));

RPF_check_toolboxes;   % check RPF dependencies

%% 1. Load data & basic fields ------------------------------------------
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

% Basic fields (all original trials)
coh_all       = allStruct.rdm1_coh(:);        % coherence
resp_all      = allStruct.req_resp(:);        % 1 = right, 2 = left
correct_all   = allStruct.correct(:);         % 1 = correct, 0 = incorrect
confCont_all  = allStruct.confidence(:);      % continuous confidence (0–1)
vol_all       = allStruct.rdm1_coh_std(:);    % stimulus-level volatility
subjID_all    = allStruct.group(:);           % subject index (1/2/3)
ME_cell_all   = allStruct.motion_energy;      % N x 1 cell

% Build a single valid mask (used for both RPF and kernels)
valid = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
         ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all);

% Restrict all fields to valid trials
coh           = coh_all(valid);
resp          = resp_all(valid);
Correct       = correct_all(valid);        % used later
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = ME_cell_all(valid);        % keep cell array aligned

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% Binary confidence: high/low (>= 0.5 = high)
Conf_raw = double(confCont);
th       = 0.5;
Conf     = double(Conf_raw >= th);         % high=1, low=0

% Coherence vector (kept for possible later use)
coh_vec  = coh;

%% 2. Map volatility to condition index (for RPF) -----------------------
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end

cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;   % low volatility
cond(vol == max(vol_levels)) = 2;   % high volatility

%% 3. Run RPF to generate p_perf(j) for each trial -----------------------
% p_perf(j) = predicted P(correct) from the RPF model

subj_list   = unique(subjID);
nSubj       = numel(subj_list);
p_perf_all  = nan(nTrials, 1);   % predicted accuracy for all valid trials

for iSub = 1:nSubj

    thisSub = subj_list(iSub);
    fprintf('\n=============================\n');
    fprintf('Running RPF for subject %d\n', thisSub);
    fprintf('=============================\n');

    % Trial index for this subject (within the valid-trial space)
    idxS = (subjID == thisSub);

    coh_s     = coh(idxS);
    resp_s    = resp(idxS);           % 1/2
    correct_s = Correct(idxS);        % 0/1
    conf_s    = confCont(idxS);       % continuous confidence
    cond_s    = cond(idxS);

    if isempty(coh_s)
        warning('Subject %d has no trials. Skipping.', thisSub);
        continue;
    end

    nTr = numel(coh_s);

    % resp 1/2 -> 0/1
    resp01 = resp_s - 1;   % 1 -> 0, 2 -> 1

    % True stimulus:
    stim01 = resp01;
    wrong_idx = (correct_s == 0);
    stim01(wrong_idx) = 1 - resp01(wrong_idx);

    % Map continuous (0–1) confidence to 4 rating levels (1..4)
    conf_clip = conf_s;
    conf_clip(conf_clip < 0) = 0;
    conf_clip(conf_clip > 1) = 1;
    edges     = [0, 0.25, 0.5, 0.75, 1];
    rating_s  = discretize(conf_clip, edges, 'IncludedEdge', 'right');
    rating_s(isnan(rating_s)) = 4;

    condition_s = cond_s;   % 1=low vol, 2=high vol

    % RPF trialData struct
    trialData = struct();
    trialData.stimID    = stim01(:)';       % 0/1
    trialData.response  = resp01(:)';       % 0/1
    trialData.rating    = rating_s(:)';     % 1..4
    trialData.correct   = correct_s(:)';    % 0/1
    trialData.x         = coh_s(:)';        % coherence
    trialData.condition = condition_s(:)';  % 1/2
    trialData.RT        = nan(1, nTr);

    % ---- F1: Performance PF d'(coh) ----
    F1 = struct();
    F1.info.DV                     = 'd''';
    F1.info.PF                     = @RPF_scaled_Weibull;
    F1.info.padCells               = 1;
    F1.info.set_P_max_to_d_pad_max = 1;
    F1.info.x_min                  = 0;
    F1.info.x_max                  = 1;
    F1.info.x_label                = 'coherence';
    F1.info.cond_labels            = {'low volatility', 'high volatility'};
    F1 = RPF_get_F(F1.info, trialData);

    % (Optional) F2 (not used)
    F2 = struct();
    F2.info.DV          = 'p(high rating)';
    F2.info.DV_respCond = 'all';
    F2.info.PF          = @PAL_Weibull;
    F2.info.x_min       = 0;
    F2.info.x_max       = 1;
    F2.info.x_label     = 'coherence';
    F2.info.cond_labels = {'low volatility', 'high volatility'};
    F2.info.constrain   = [];
    F2 = RPF_get_F(F2.info, trialData);

    % ---- Use F1 to generate p_perf_trial ----
    p_perf_trial = nan(nTr, 1);
    nCond = numel(F1.data);

    for c = 1:nCond
        mask_c = (condition_s == c);
        if ~any(mask_c), continue; end

        coh_c  = coh_s(mask_c);
        x_grid = F1.data(c).x(:);
        d_grid = F1.data(c).P(:);

        [~, loc] = ismember(coh_c, x_grid);

        d_pred = nan(size(coh_c));
        ok = (loc > 0);
        d_pred(ok) = d_grid(loc(ok));

        if any(~ok)
            d_pred(~ok) = interp1(x_grid, d_grid, coh_c(~ok), 'linear', 'extrap');
        end

        p_corr = normcdf(d_pred ./ sqrt(2));
        p_perf_trial(mask_c) = p_corr;
    end

    p_perf_all(idxS) = p_perf_trial;
end

fprintf('Finished RPF. Valid p_perf proportion: %.3f\n', mean(~isnan(p_perf_all)));

%% 4. Compute residual volatility from motion_energy ---------------------
winLen = 10;
tol    = 1e-12;

evidence_strength   = cell(nTrials, 1);
volatility_strength = cell(nTrials, 1);

for tr = 1:nTrials
    frames = motion_energy{tr};
    trace  = frames(:)';

    last_nz = find(abs(trace) > tol, 1, 'last');
    if isempty(last_nz)
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);

    if nFrames < winLen
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    nWin  = nFrames - winLen + 1;
    m_win = nan(1, nWin);
    s_win = nan(1, nWin);

    for w = 1:nWin
        seg      = trace_eff(w : w + winLen - 1);
        m_win(w) = mean(seg);
        s_win(w) = std(seg);
    end

    evidence_strength{tr}   = m_win;
    volatility_strength{tr} = s_win;
end

% Warp to normalized time axis
nBins  = 40;
t_norm = linspace(0, 1, nBins);

MEAN_norm = nan(nTrials, nBins);
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};

    if isempty(mu_tr) || isempty(sd_tr), continue; end

    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    t_orig = linspace(0, 1, nWin_tr);

    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

% Residual volatility per bin: STD ~ |MEAN|
resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 3, continue; end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    y_hat = Xb * beta;
    resid = y_use - y_hat;

    tmp = nan(size(y));
    tmp(mask_b) = resid;
    resVol_mat(:, b) = tmp;
end

% Global z-score
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_mat = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d time bins.\n', size(resVol_mat,1), size(resVol_mat,2));
resVol_time = resVol_mat;

%% 5. Build performance predictor f(p_j) ---------------------------------
eps0 = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);

f_perf = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

%% 6. Fit M0–M8 per time bin (fitglm) + store AIC/BIC + betas -----------
[N, K] = size(resVol_time);
assert(K == 40, 'Expected K=40 time bins.');

minN = 50;
useSubjDummies = true;

Fp_all = f_perf;
Fp_all = (Fp_all - mean(Fp_all,'omitnan')) ./ std(Fp_all,'omitnan');
Cz_all = Correct - mean(Correct,'omitnan');

modelNames = {'M0','M1','M2','M3','M4','M5','M6','M7','M8'};

% extras in order: [PxC, PxV, VxC, PxVxC]
modelExtras = {
    [0 0 0 0]
    [1 0 0 0]
    [0 1 0 0]
    [0 0 1 0]
    [1 1 0 0]
    [1 0 1 0]
    [0 1 1 0]
    [1 1 1 0]
    [1 1 1 1]
};

baseLabels  = {'b0 (Intercept)','b_{perf}','b_{corr}','b_{vol}'};
extraLabels = {'b_{perf×corr}','b_{perf×vol}','b_{vol×corr}','b_{perf×vol×corr}'};

AIC_mat  = nan(K, numel(modelNames));
BIC_mat  = nan(K, numel(modelNames));
Nobs_mat = nan(K, numel(modelNames));

Models = struct();

for m = 1:numel(modelNames)

    use_PxC   = modelExtras{m}(1);
    use_PxV   = modelExtras{m}(2);
    use_VxC   = modelExtras{m}(3);
    use_PxVxC = modelExtras{m}(4);

    labels = baseLabels;
    if use_PxC,   labels{end+1} = extraLabels{1}; end
    if use_PxV,   labels{end+1} = extraLabels{2}; end
    if use_VxC,   labels{end+1} = extraLabels{3}; end
    if use_PxVxC, labels{end+1} = extraLabels{4}; end

    nTerms = numel(labels);
    betas  = nan(K, nTerms);

    fprintf('\n=== Fitting %s (fitglm) ===\n', modelNames{m});

    for k = 1:K
        Vk = resVol_time(:,k);

        mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(subjID);
        if sum(mask) < minN, continue; end

        y    = Conf(mask);
        C    = Cz_all(mask);
        P    = Fp_all(mask);
        Vraw = Vk(mask);
        sID  = subjID(mask);

        sv = std(Vraw);
        if sv < 1e-12, continue; end
        V = (Vraw - mean(Vraw)) ./ sv;

        PxC   = P .* C;
        PxV   = P .* V;
        VxC   = V .* C;
        PxVxC = P .* V .* C;

        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);
            T = table(y, P, C, V, PxC, PxV, VxC, PxVxC, S2, S3, ...
                'VariableNames', {'conf','perf','corr','vol','PxC','PxV','VxC','PxVxC','S2','S3'});
        else
            T = table(y, P, C, V, PxC, PxV, VxC, PxVxC, ...
                'VariableNames', {'conf','perf','corr','vol','PxC','PxV','VxC','PxVxC'});
        end

        f = "conf ~ perf + corr + vol";
        if use_PxC,   f = f + " + PxC";   end
        if use_PxV,   f = f + " + PxV";   end
        if use_VxC,   f = f + " + VxC";   end
        if use_PxVxC, f = f + " + PxVxC"; end
        if useSubjDummies, f = f + " + S2 + S3"; end

        try
            g = fitglm(T, f, 'Distribution','binomial', 'Link','logit');
        catch
            continue;
        end

        AIC_mat(k,m)  = g.ModelCriterion.AIC;
        BIC_mat(k,m)  = g.ModelCriterion.BIC;
        Nobs_mat(k,m) = sum(mask);

        coefNames = string(g.CoefficientNames);
        coefVals  = g.Coefficients.Estimate;
        
        pVals = g.Coefficients.pValue;
        
        getb = @(nm) coefVals(find(coefNames==nm,1,'first'));
        %get p "pVals = g.Coefficients.pValue;"
        betas(k,1) = getb("(Intercept)");
        betas(k,2) = getb("perf");
        betas(k,3) = getb("corr");
        betas(k,4) = getb("vol");

        if use_PxC
            betas(k, find(strcmp(labels, extraLabels{1}))) = getb("PxC");
            %p_val
        end
        if use_PxV
            betas(k, find(strcmp(labels, extraLabels{2}))) = getb("PxV");
        end
        if use_VxC
            betas(k, find(strcmp(labels, extraLabels{3}))) = getb("VxC");
        end
        if use_PxVxC
            betas(k, find(strcmp(labels, extraLabels{4}))) = getb("PxVxC");
        end
    end

    Models(m).name   = modelNames{m};
    Models(m).labels = labels;
    Models(m).betas  = betas;
end

%% Delta AIC/BIC summary (NO win-bins) -----------------------------------
minAIC_perBin = min(AIC_mat, [], 2, 'omitnan');
minBIC_perBin = min(BIC_mat, [], 2, 'omitnan');

deltaAIC_mat = AIC_mat - minAIC_perBin;
deltaBIC_mat = BIC_mat - minBIC_perBin;

meanDeltaAIC = mean(deltaAIC_mat, 1, 'omitnan');
medDeltaAIC  = median(deltaAIC_mat, 1, 'omitnan');
meanDeltaBIC = mean(deltaBIC_mat, 1, 'omitnan');
medDeltaBIC  = median(deltaBIC_mat, 1, 'omitnan');

deltaTbl = table(modelNames(:), meanDeltaAIC(:), medDeltaAIC(:), ...
                             meanDeltaBIC(:), medDeltaBIC(:), ...
    'VariableNames', {'Model','Mean_dAIC','Median_dAIC','Mean_dBIC','Median_dBIC'});

disp('=== Time-resolved Delta AIC/BIC summary (per model) ===');
disp(deltaTbl);

%% === 6.5 FIGURE TABLE: white background + subtle highlight winners =========
winningModels = {'M1','M4','M5','M7','M8'};
T = deltaTbl;

% Convert to cell
C = table2cell(T);
colNames = T.Properties.VariableNames;

% Subtle highlight: slightly darker text (not bold)
% You can tweak highlightColor from 0 to 1: smaller = darker
highlightColor = '#111111';   % almost black
normalColor    = '#666666';   % dark gray (less contrast)
monoFont = 'Courier New';

toHTML = @(x, col) ['<html><span style="color:' col '; font-family:' monoFont ';">' ...
                    char(string(x)) '</span></html>'];

for i = 1:size(C,1)
    thisModel = string(C{i,1});
    isWin = ismember(thisModel, string(winningModels));

    for j = 1:size(C,2)
        if isWin
            C{i,j} = toHTML(C{i,j}, highlightColor);
        else
            C{i,j} = toHTML(C{i,j}, normalColor);
        end
    end
end

% Make figure/uitable
figTbl = figure('Color','w','Name','Model comparison (Delta AIC/BIC)');
t = uitable(figTbl, ...
    'Data', C, ...
    'ColumnName', colNames, ...
    'RowName', [], ...
    'Units','normalized', ...
    'Position',[0 0 1 1]);

% Fonts & sizing
t.FontName = monoFont;
t.FontSize = 14;

% Column widths (tweak as needed)
t.ColumnWidth = {90, 140, 150, 140, 150};

% ---- Force white background (remove grey zebra striping) ----
% Works in many MATLAB versions; if your version ignores it, see note below.
t.BackgroundColor = [1 1 1];   % pure white

% Optional title
annotation(figTbl,'textbox',[0 0.93 1 0.06], ...
    'String','Time-resolved Delta AIC/BIC summary (winners subtly emphasized)', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontWeight','bold','FontSize',16);

% Export
% --- R2019a export replacement ---
set(figTbl,'Renderer','painters');   % good for pdf
print(figTbl, 'ModelComparisonTable_Subtle', '-dpdf', '-painters');  % PDF

set(figTbl,'Renderer','opengl');     % good for raster png
print(figTbl, 'ModelComparisonTable_Subtle', '-dpng', '-r300');     % PNG 300dpi


fprintf('Saved: ModelComparisonTable_Subtle.png / .pdf\n');



%% 7. Plot: per-model small figures (DISABLED) --------------------------
if DO_PLOT_SECTION7
    allVals = [];
    for m = 1:numel(Models)
        tmp = Models(m).betas;
        allVals = [allVals; tmp(:)];
    end
    allVals = allVals(~isnan(allVals));
    if isempty(allVals), error('No betas estimated.'); end
    yMin = min(allVals); yMax = max(allVals);
    pad  = 0.05 * (yMax - yMin + eps);
    yLimGlobal_tmp = [yMin - pad, yMax + pad];

    for m = 1:numel(Models)
        bet   = Models(m).betas;
        labs  = Models(m).labels;
        nSub  = numel(labs);

        figure('Name', Models(m).name, 'Color', 'w');
        for s = 1:nSub
            subplot(nSub,1,s);
            plot(t_norm, bet(:,s), '-o', 'LineWidth', 1.2); hold on;
            yline(0,'k--');
            xlim([0 1]);
            ylim(yLimGlobal_tmp);
            grid on;
            xlabel('Normalized time');
            ylabel(labs{s});
            title(sprintf('%s: %s', Models(m).name, labs{s}), 'Interpreter','none');
        end
    end
end

%% 8. FORCE models to plot: M1 M4 M5 M7 M8 ------------------------------
% modelNames = {'M0','M1','M2','M3','M4','M5','M6','M7','M8'};
modelIdxToPlot   = [2 5 6 8 9];  % M1 M4 M5 M7 M8
modelNamesToPlot = modelNames(modelIdxToPlot);
fprintf('\n=== Models to plot (FORCED ORDER) ===\n');
disp(modelNamesToPlot);

%% 9) Refit per subject per bin for selected models (STORE Sel: beta/SE/p) ----
subj_list = unique(subjID(:))';
nSubj = numel(subj_list);
K = numel(t_norm);

colSub = [
    0 0.4470 0.7410
    1 0 0
    0.9290 0.6940 0.1250
];

minN_sub = 5;
sv_tol   = 1e-12;

Sel = struct();

for ii = 1:numel(modelIdxToPlot)

    mIdx   = modelIdxToPlot(ii);
    mName  = modelNames{mIdx};

    use_PxC   = modelExtras{mIdx}(1);
    use_PxV   = modelExtras{mIdx}(2);
    use_VxC   = modelExtras{mIdx}(3);
    use_PxVxC = modelExtras{mIdx}(4);

    termNames  = ["(Intercept)","perf","corr","vol"];
    termLabels = {'b0 (Intercept)','b_{perf}','b_{corr}','b_{vol}'};

    if use_PxC
        termNames(end+1)  = "PxC";
        termLabels{end+1} = 'b_{perf×corr}';
    end
    if use_PxV
        termNames(end+1)  = "PxV";
        termLabels{end+1} = 'b_{perf×vol}';
    end
    if use_VxC
        termNames(end+1)  = "VxC";
        termLabels{end+1} = 'b_{vol×corr}';
    end
    if use_PxVxC
        termNames(end+1)  = "PxVxC";
        termLabels{end+1} = 'b_{perf×vol×corr}';
    end

    nTerms = numel(termNames);

    beta_sub = nan(nSubj, K, nTerms);
    se_sub   = nan(nSubj, K, nTerms);
    p_sub    = nan(nSubj, K, nTerms);   % NEW: store p-values from fitglm

    fprintf('\n--- Refit %s per subject/per bin ---\n', mName);

    for iSub = 1:nSubj
        thisSub = subj_list(iSub);

        for k = 1:K
            Vk = resVol_time(:,k);

            mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Correct) & ~isnan(Fp_all) & ~isnan(subjID);
            mask = mask & (subjID == thisSub);

            if sum(mask) < minN_sub, continue; end

            y    = Conf(mask);
            C    = Cz_all(mask);
            P    = Fp_all(mask);
            Vraw = Vk(mask);

            sv = std(Vraw);
            if sv < sv_tol, continue; end
            V = (Vraw - mean(Vraw)) ./ sv;

            PxC   = P .* C;
            PxV   = P .* V;
            VxC   = V .* C;
            PxVxC = P .* V .* C;

            T = table(y, P, C, V, PxC, PxV, VxC, PxVxC, ...
                'VariableNames', {'conf','perf','corr','vol','PxC','PxV','VxC','PxVxC'});

            f = "conf ~ perf + corr + vol";
            if use_PxC,   f = f + " + PxC";   end
            if use_PxV,   f = f + " + PxV";   end
            if use_VxC,   f = f + " + VxC";   end
            if use_PxVxC, f = f + " + PxVxC"; end

            try
                g = fitglm(T, f, 'Distribution','binomial', 'Link','logit');
            catch
                continue;
            end

            coefNames = string(g.CoefficientNames);
            coefEst   = g.Coefficients.Estimate;
            coefSE    = g.Coefficients.SE;
            coefP     = g.Coefficients.pValue;

            for tt = 1:nTerms
                nm  = termNames(tt);
                idx = find(coefNames == nm, 1, 'first');
                if ~isempty(idx)
                    beta_sub(iSub,k,tt) = coefEst(idx);
                    se_sub(iSub,k,tt)   = coefSE(idx);
                    p_sub(iSub,k,tt)    = coefP(idx);
                end
            end
        end
    end

    Sel(ii).mName      = mName;
    Sel(ii).termLabels = termLabels;
    Sel(ii).beta_sub   = beta_sub;
    Sel(ii).se_sub     = se_sub;
    Sel(ii).p_sub      = p_sub;
end

% ===== global y-limits using ALL selected models/terms/subjects/time =====
allY = [];
for ii = 1:numel(Sel)
    b = Sel(ii).beta_sub;
    e = Sel(ii).se_sub;
    allY = [allY; b(:); (b(:)-e(:)); (b(:)+e(:))];
end
allY = allY(~isnan(allY));
if isempty(allY)
    error('No beta/SE values available. Check masks/minN_sub.');
end

yMin = min(allY);
yMax = max(allY);
pad  = 0.05 * (yMax - yMin + eps);
yLimGlobal = [yMin - pad, yMax + pad];

fprintf('\nGlobal y-limits for BIG FIG: [%.3f, %.3f]\n', yLimGlobal(1), yLimGlobal(2));

% ---- PASS 2 plots (disabled) ----
if DO_PLOT_SECTION9_PASS2
    % (intentionally off)
end



%% 11) DIRECTIONAL SIGNIFICANCE PREVALENCE (MODEL × SUBJECT) USING p-values ONLY
% =========================
% significant if p < alpha (two-sided), direction from sign(beta)

alpha = 0.05;  % two-sided p threshold

masterTermLabels = {'b0 (Intercept)','b_{perf}','b_{corr}','b_{vol}', ...
                    'b_{perf×corr}','b_{perf×vol}','b_{vol×corr}','b_{perf×vol×corr}'};

K            = numel(t_norm);
nModelsSel   = numel(Sel);
nTermsMaster = numel(masterTermLabels);

SigPrev = struct();
SigPrev.alpha      = alpha;
SigPrev.termLabels = masterTermLabels;
SigPrev.modelNames = string({Sel.mName});

SigPrev.denom     = nan(K, nTermsMaster);
SigPrev.posCount  = nan(K, nTermsMaster);
SigPrev.negCount  = nan(K, nTermsMaster);

SigPrev.posFrac   = nan(K, nTermsMaster);
SigPrev.negFrac   = nan(K, nTermsMaster);
SigPrev.allFrac   = nan(K, nTermsMaster);

for r = 1:nTermsMaster
    wantLabel = masterTermLabels{r};

    for k = 1:K
        den_k = 0;
        pos_k = 0;
        neg_k = 0;

        for m = 1:nModelsSel
            termList = Sel(m).termLabels;
            tt = find(strcmp(termList, wantLabel), 1, 'first');
            if isempty(tt)
                continue; % model doesn't have this term
            end

            beta_sub = Sel(m).beta_sub(:,:,tt); % [nSubj × K]
            p_sub    = Sel(m).p_sub(:,:,tt);    % [nSubj × K]

            for s = 1:nSubj
                b = beta_sub(s,k);
                p = p_sub(s,k);

                if isnan(b) || isnan(p), continue; end

                den_k = den_k + 1;

                if p < alpha
                    if b > 0
                        pos_k = pos_k + 1;
                    elseif b < 0
                        neg_k = neg_k + 1;
                    end
                end
            end
        end

        SigPrev.denom(k,r)    = den_k;
        SigPrev.posCount(k,r) = pos_k;
        SigPrev.negCount(k,r) = neg_k;

        if den_k > 0
            SigPrev.posFrac(k,r) = pos_k / den_k;
            SigPrev.negFrac(k,r) = neg_k / den_k;
            SigPrev.allFrac(k,r) = (pos_k + neg_k) / den_k;
        else
            SigPrev.posFrac(k,r) = NaN;
            SigPrev.negFrac(k,r) = NaN;
            SigPrev.allFrac(k,r) = NaN;
        end
    end
end


%% 12) PLOT: Significance prevalence (overall/pos/neg) --------------------
figSig = figure('Color','w','Name','Directional prevalence across Model×Subject');

for r = 1:nTermsMaster
    ax = subplot(4,2,r);
    hold(ax,'on'); grid(ax,'on');

    yAll = SigPrev.allFrac(:,r);
    yPos = SigPrev.posFrac(:,r);
    yNeg = SigPrev.negFrac(:,r);
    yDen = SigPrev.denom(:,r);

    plot(ax, t_norm, yAll, 'k-', 'LineWidth', 1.8);
    plot(ax, t_norm, yPos, 'r-', 'LineWidth', 1.0);
    plot(ax, t_norm, yNeg, 'b-', 'LineWidth', 1.0);

    ylim(ax, [0 1]);
    xlim(ax, [0 1]);
    set(ax, 'XTick', 0:0.2:1);

    ylabel(ax, 'Fraction significant', 'Interpreter','latex');

    % Use the SAME LaTeX term labels as in BIG FIG
    latexTermLabels = { ...
        '$b_0$ (Intercept)', ...
        '$b_{\mathrm{perf}}$', ...
        '$b_{\mathrm{corr}}$', ...
        '$b_{\mathrm{vol}}$', ...
        '$b_{\mathrm{perf}\times\mathrm{corr}}$', ...
        '$b_{\mathrm{perf}\times\mathrm{vol}}$', ...
        '$b_{\mathrm{vol}\times\mathrm{corr}}$', ...
        '$b_{\mathrm{perf}\times\mathrm{vol}\times\mathrm{corr}}$' ...
    };
    title(ax, latexTermLabels{r}, 'Interpreter','latex');

    % ===== change "max denom" text =====
    maxDen = max(yDen, [], 'omitnan');
    if isempty(maxDen) || isnan(maxDen), maxDen = 0; end

    txt = sprintf('$\\#\\,\\mathrm{valid\\ fits\\ (Model\\times Subject)}\\,=\\,%g$', maxDen);

    text(ax, 0.02, 0.90, txt, 'Units','normalized', 'FontSize', 8, 'Interpreter','latex');

    if r == 1
        legend(ax, {'overall','positive','negative'}, ...
            'Location','southoutside', 'Orientation','horizontal', ...
            'Box','off', 'Interpreter','latex');
    end
end

% ===== Figure-level title (LaTeX, robust) =====
mainTitle = sprintf(['Directional prevalence of non-zero betas ', ...
                     '(fitglm $p$-values, two-sided $\\alpha = %.3f$)'], alpha);

% Prefer sgtitle if available (supports Interpreter='latex' reliably in newer MATLAB)
if exist('sgtitle','file') == 2
    sgtitle(figSig, mainTitle, 'Interpreter','latex', 'FontWeight','bold');
else
    % Fallback: annotation (works everywhere)
    annotation(figSig,'textbox',[0 0.96 1 0.04], ...
        'String', mainTitle, ...
        'EdgeColor','none', ...
        'HorizontalAlignment','center', ...
        'FontWeight','bold', ...
        'Interpreter','latex');
end
