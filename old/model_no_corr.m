%% no corr

clear; clc;
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

    % ---- Build RPF trialData ----

    % resp 1/2 -> 0/1
    resp01 = resp_s - 1;   % 1 -> 0, 2 -> 1

    % True stimulus:
    % if correct: stim = resp; if wrong: stim = 1 - resp
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

    % condition: 1 = low vol, 2 = high vol
    condition_s = cond_s;

    % RPF trialData struct
    trialData = struct();
    trialData.stimID    = stim01(:)';       % 1×nTr, 0/1
    trialData.response  = resp01(:)';       % 1×nTr, 0/1
    trialData.rating    = rating_s(:)';     % 1×nTr, 1..4
    trialData.correct   = correct_s(:)';    % 1×nTr, 0/1
    trialData.x         = coh_s(:)';        % 1×nTr, coherence
    trialData.condition = condition_s(:)';  % 1×nTr, 1/2
    trialData.RT        = nan(1, nTr);      % no RT

    % ---- F1: Performance psychometric function d'(coh) ----
    F1 = struct();
    F1.info.DV                     = 'd''';
    F1.info.PF                     = @RPF_scaled_Weibull;
    F1.info.padCells               = 1;
    F1.info.set_P_max_to_d_pad_max = 1;
    F1.info.x_min                  = 0;
    F1.info.x_max                  = 1;
    F1.info.x_label                = 'coherence';
    F1.info.cond_labels            = {'low volatility', 'high volatility'};

    F1 = RPF_get_F(F1.info, trialData);   % fit performance PF

    % ---- (Optional) F2: Confidence PF (not used later, kept for structure) ----
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

    % ---- Use F1 to generate p_perf_trial for each trial ----
    p_perf_trial = nan(nTr, 1);   % trialwise P(correct)

    nCond = numel(F1.data);       % usually 2 conditions: low & high vol

    for c = 1:nCond
        % Trials in this volatility condition
        mask_c = (condition_s == c);
        if ~any(mask_c)
            continue;
        end

        coh_c = coh_s(mask_c);         % coherence for these trials
        x_grid = F1.data(c).x(:);      % coherence grid
        d_grid = F1.data(c).P(:);      % predicted d'(x)

        % Try exact matching coh_c to x_grid
        % Try exact matching coh_c to x_grid
        [~, loc] = ismember(coh_c, x_grid);

        % Preallocate
        d_pred = nan(size(coh_c));

        % 1) Exact matches: lookup from grid
        ok = (loc > 0);
        d_pred(ok) = d_grid(loc(ok));

        % 2) Unmatched values: interpolate (handle float mismatch / out-of-grid)
        if any(~ok)
            d_pred(~ok) = interp1(x_grid, d_grid, coh_c(~ok), 'linear', 'extrap');
        end


        % Convert d' -> P(correct) for 2AFC
        p_corr = normcdf(d_pred ./ sqrt(2));

        p_perf_trial(mask_c) = p_corr;
    end

    % Write back into the full p_perf_all vector
    p_perf_all(idxS) = p_perf_trial;

end

fprintf('Finished RPF. Valid p_perf proportion: %.3f\n', ...
        mean(~isnan(p_perf_all)));

%% 4. Compute residual volatility from motion_energy ---------------------
% Steps: sliding-window mean/std + warp to normalized time + regress STD ~ |MEAN|

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

% Warp to 0–1 normalized time axis
nBins  = 40;
t_norm = linspace(0, 1, nBins);

MEAN_norm = nan(nTrials, nBins);
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};

    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end

    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    t_orig = linspace(0, 1, nWin_tr);

    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

% For each time bin: STD ~ |MEAN| -> residual volatility
resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 3
        continue;
    end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    y_hat = Xb * beta;
    resid = y_use - y_hat;

    tmp       = nan(size(y));
    tmp(mask_b) = resid;
    resVol_mat(:, b) = tmp;
end

% Global z-score (scaling only)
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_mat = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d time bins.\n', ...
        size(resVol_mat,1), size(resVol_mat,2));

resVol_time = resVol_mat;   % N x K

%% 5. Build performance predictor f(p_j) ---------------------------------
% p_perf_all: RPF predicted accuracy per trial (0~1)

eps = 1e-4;
p_clip = min(max(p_perf_all, eps), 1-eps);   % avoid 0 or 1

% Simple choice: z-score as f(p_j)
f_perf = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

%% 6. Fit stimulus-only models per time bin (fitglm) + print average AIC/BIC ----
% Models WITHOUT corr:
%   M0: conf ~ perf + vol (+ subj dummies)
%   M1: conf ~ perf + vol + perf×vol (+ subj dummies)
%% Define stimulus-only model family (NO corr)

modelNames = {'M0','M1','M2','M3','M4'};

% modelFlags: [usePerf, useVol, usePerf×Vol]
modelFlags = {
    [0 0 0]   % M0: intercept only
    [1 0 0]   % M1: perf
    [0 1 0]   % M2: vol
    [1 1 0]   % M3: perf + vol
    [1 1 1]   % M4: perf + vol + perf×vol
};

[N, K] = size(resVol_time);
assert(K == 40, 'Expected K=40 time bins.');

minN = 50;
useSubjDummies = true;

% Make sure perf predictor is z-scored
Fp_all = f_perf;
Fp_all = (Fp_all - mean(Fp_all,'omitnan')) ./ std(Fp_all,'omitnan');


% extras in order: [PxV]
%modelExtras = {
%    0
%    1
%};

baseLabels  = {'b0 (Intercept)','b_{perf}','b_{vol}'};
extraLabels = {'b_{perf×vol}'};

% Store per-bin criteria
AIC_mat  = nan(K, numel(modelNames));
BIC_mat  = nan(K, numel(modelNames));
Nobs_mat = nan(K, numel(modelNames));

Models = struct();

for m = 1:numel(modelNames)

    useP   = modelFlags{m}(1);
    useV   = modelFlags{m}(2);
    usePxV = modelFlags{m}(3);


    labels = baseLabels;
    if usePxV, labels{end+1} = extraLabels{1}; end

    nTerms = numel(labels);
    betas  = nan(K, nTerms);

    fprintf('\n=== Fitting %s (stimulus-only, fitglm) ===\n', modelNames{m});

    for k = 1:K
        Vk = resVol_time(:,k);

        mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Fp_all) & ~isnan(subjID);
        if sum(mask) < minN
            continue;
        end

        y    = Conf(mask);
        P    = Fp_all(mask);
        Vraw = Vk(mask);
        sID  = subjID(mask);

        % z-score V within this bin
        sv = std(Vraw);
        if sv < 1e-12
            continue;
        end
        V = (Vraw - mean(Vraw)) ./ sv;

        % interaction term
        PxV = P .* V;

        % Build table for fitglm
        if useSubjDummies
            S2 = double(sID == 2);
            S3 = double(sID == 3);
            T = table(y, P, V, PxV, S2, S3, ...
                'VariableNames', {'conf','perf','vol','PxV','S2','S3'});
        else
            T = table(y, P, V, PxV, ...
                'VariableNames', {'conf','perf','vol','PxV'});
        end

        % Build formula string
        f = "conf ~ perf + vol";
        if usePxV, f = f + " + PxV"; end
        if useSubjDummies
            f = f + " + S2 + S3";
        end

        % Fit
        try
            g = fitglm(T, f, 'Distribution','binomial', 'Link','logit');
        catch
            continue;
        end

        % store AIC/BIC
        AIC_mat(k,m)  = g.ModelCriterion.AIC;
        BIC_mat(k,m)  = g.ModelCriterion.BIC;
        Nobs_mat(k,m) = sum(mask);

        % store betas
        coefNames = string(g.CoefficientNames);
        coefVals  = g.Coefficients.Estimate;
        getb = @(nm) coefVals(find(coefNames==nm,1,'first'));

        betas(k,1) = getb("(Intercept)");
        betas(k,2) = getb("perf");
        betas(k,3) = getb("vol");

        if usePxV
            betas(k, find(strcmp(labels, extraLabels{1}))) = getb("PxV");
        end
    end

    Models(m).name   = modelNames{m};
    Models(m).labels = labels;
    Models(m).betas  = betas;
end

%% Print average AIC/BIC per model across time bins ----------------------
meanAIC = mean(AIC_mat, 1, 'omitnan');
meanBIC = mean(BIC_mat, 1, 'omitnan');
medAIC  = median(AIC_mat, 1, 'omitnan');
medBIC  = median(BIC_mat, 1, 'omitnan');
nBinsUsed = sum(~isnan(AIC_mat), 1);

critTbl = table(modelNames(:), nBinsUsed(:), meanAIC(:), medAIC(:), meanBIC(:), medBIC(:), ...
    'VariableNames', {'Model','nTimeBins','MeanAIC','MedianAIC','MeanBIC','MedianBIC'});

disp('=== AIC/BIC summary across time bins (stimulus-only) ===');
disp(critTbl);

%% 7. Plot: one figure per model + GLOBAL same y-limits
allVals = [];
for m = 1:numel(Models)
    tmp = Models(m).betas;
    allVals = [allVals; tmp(:)];
end
allVals = allVals(~isnan(allVals));
yMin = min(allVals); yMax = max(allVals);
pad  = 0.05 * (yMax - yMin + eps);
yLimGlobal = [yMin - pad, yMax + pad];

for m = 1:numel(Models)
    bet  = Models(m).betas;
    labs = Models(m).labels;
    nSub = numel(labs);

    figure('Name', Models(m).name, 'Color', 'w');
    for s = 1:nSub
        subplot(nSub,1,s);
        plot(t_norm, bet(:,s), '-o', 'LineWidth', 1.2); hold on;
        yline(0,'k--'); xlim([0 1]); ylim(yLimGlobal);
        grid on;
        xlabel('Normalized time');
        ylabel(labs{s});
        title(sprintf('%s: %s', Models(m).name, labs{s}), 'Interpreter','none');
    end
end

%% 8) Pick models to plot
% stimulus-only family: plot ALL models (M0..M4)
modelIdxToPlot  = 1:numel(modelNames);
modelNamesToPlot = modelNames(modelIdxToPlot);

fprintf('\n=== Models to plot (stimulus-only) ===\n');
disp(modelNamesToPlot(:));


%% 9) For each selected model: refit per subject per bin, then plot each beta with SE bands
subj_list = unique(subjID(:))';
nSubj = numel(subj_list);
K = numel(t_norm);

% colors (match your previous scheme)
colSub = [
    0 0.4470 0.7410
    1 0 0
    0.9290 0.6940 0.1250
];

minN_sub = 5;
sv_tol   = 1e-12;

% ========= PASS 1: Refit & STORE (do NOT plot yet) =========
Sel = struct();

for ii = 1:numel(modelIdxToPlot)

    mIdx  = modelIdxToPlot(ii);
    mName = modelNames{mIdx};

    % flags: [useP useV usePxV]
    useP   = modelFlags{mIdx}(1);
    useV   = modelFlags{mIdx}(2);
    usePxV = modelFlags{mIdx}(3);

    % --- dynamic term list for this model ---
    termNames  = "(Intercept)";
    termLabels = {'b0 (Intercept)'};

    if useP
        termNames(end+1)  = "perf";
        termLabels{end+1} = 'b_{perf}';
    end
    if useV
        termNames(end+1)  = "vol";
        termLabels{end+1} = 'b_{vol}';
    end
    if usePxV
        termNames(end+1)  = "PxV";
        termLabels{end+1} = 'b_{perf×vol}';
    end

    nTerms = numel(termNames);

    beta_sub = nan(nSubj, K, nTerms);
    se_sub   = nan(nSubj, K, nTerms);

    fprintf('\n--- Refit %s per subject/per bin (stimulus-only) ---\n', mName);

    for iSub = 1:nSubj
        thisSub = subj_list(iSub);

        for k = 1:K
            Vk = resVol_time(:,k);

            % build mask depending on whether model uses P/V
            mask = ~isnan(Conf) & ~isnan(subjID) & (subjID == thisSub);
            if useP, mask = mask & ~isnan(Fp_all); end
            if useV, mask = mask & ~isnan(Vk);     end

            if sum(mask) < minN_sub
                continue;
            end

            y = Conf(mask);

            % perf
            if useP
                P = Fp_all(mask);
            else
                P = nan(size(y));
            end

            % vol (z-scored within this bin, within this subject)
            if useV
                Vraw = Vk(mask);
                sv = std(Vraw);
                if sv < sv_tol
                    continue;
                end
                V = (Vraw - mean(Vraw)) ./ sv;
            else
                V = nan(size(y));
            end

            % interaction
            if usePxV
                PxV = P .* V;
            else
                PxV = nan(size(y));
            end

            % table (keep variable names stable across models)
            T = table(y, P, V, PxV, ...
                'VariableNames', {'conf','perf','vol','PxV'});

            % formula
            f = "conf ~ 1";
            if useP,   f = f + " + perf"; end
            if useV,   f = f + " + vol";  end
            if usePxV, f = f + " + PxV";  end

            % fit
            try
                g = fitglm(T, f, 'Distribution','binomial', 'Link','logit');
            catch
                continue;
            end

            coefNames = string(g.CoefficientNames);
            coefEst   = g.Coefficients.Estimate;
            coefSE    = g.Coefficients.SE;

            for tt = 1:nTerms
                nm  = termNames(tt);
                idx = find(coefNames == nm, 1, 'first');
                if ~isempty(idx)
                    beta_sub(iSub,k,tt) = coefEst(idx);
                    se_sub(iSub,k,tt)   = coefSE(idx);
                end
            end
        end
    end

    Sel(ii).mName      = mName;
    Sel(ii).termLabels = termLabels;
    Sel(ii).beta_sub   = beta_sub;
    Sel(ii).se_sub     = se_sub;
end


% ========= Reorder columns: put the model with MOST terms first =========
nTermsEach = cellfun(@numel, {Sel.termLabels});
[~, idxMax] = max(nTermsEach);
Sel = Sel([idxMax, setdiff(1:numel(Sel), idxMax, 'stable')]);


% ========= Compute GLOBAL y-limits across ALL selected models/terms/subjects/time =========
allY = [];
for ii = 1:numel(Sel)
    b = Sel(ii).beta_sub;
    e = Sel(ii).se_sub;
    allY = [allY; b(:)];
    allY = [allY; (b(:) - e(:))];
    allY = [allY; (b(:) + e(:))];
end
allY = allY(~isnan(allY));
if isempty(allY)
    error('No beta/SE values available. Check masks/minN_sub.');
end
yMin = min(allY);
yMax = max(allY);
pad  = 0.05 * (yMax - yMin + eps);
yLimGlobal = [yMin - pad, yMax + pad];

fprintf('\nGlobal y-limits for ALL plots in Section 9: [%.3f, %.3f]\n', yLimGlobal(1), yLimGlobal(2));


% ========= PASS 2: Plot each (model, term) as its own figure =========
x = t_norm(:)';

for ii = 1:numel(Sel)

    mName      = Sel(ii).mName;
    termLabels = Sel(ii).termLabels;
    beta_sub   = Sel(ii).beta_sub;
    se_sub     = Sel(ii).se_sub;

    nTerms = numel(termLabels);

    for tt = 1:nTerms
        figure('Color','w','Name',sprintf('%s: %s', mName, termLabels{tt}));
        hold on; grid on;

        hLine = gobjects(nSubj,1);

        for iSub = 1:nSubj
            yv = squeeze(beta_sub(iSub,:,tt));
            ev = squeeze(se_sub(iSub,:,tt));

            hLine(iSub) = plot(x, yv, '-', 'Color', colSub(iSub,:), 'LineWidth', 2.0);

            ok = ~isnan(yv) & ~isnan(ev);
            if sum(ok) >= 2
                xx = x(ok); yy = yv(ok); ee = ev(ok);
                fill([xx, fliplr(xx)], [yy-ee, fliplr(yy+ee)], colSub(iSub,:), ...
                    'EdgeColor','none', 'FaceAlpha',0.18, 'HandleVisibility','off');
            end
        end

        % mean ± SEM across subjects
        yMean = squeeze(mean(beta_sub(:,:,tt), 1, 'omitnan'));
        ySEM  = squeeze(std(beta_sub(:,:,tt), 0, 1, 'omitnan')) ./ sqrt(nSubj);

        okm = ~isnan(yMean) & ~isnan(ySEM);
        if sum(okm) >= 2
            xx = x(okm); ym = yMean(okm); es = ySEM(okm);
            fill([xx, fliplr(xx)], [ym-es, fliplr(ym+es)], [0 0 0], ...
                'EdgeColor','none', 'FaceAlpha',0.10, 'HandleVisibility','off');
        end
        hMean = plot(x, yMean, 'k-', 'LineWidth', 3.0);

        yline(0,'k--','HandleVisibility','off');
        xlim([0 1]); xticks(0:0.1:1);
        ylim(yLimGlobal);

        xlabel('Normalized time (0–1)');
        ylabel('Beta');
        title(sprintf('%s — %s', mName, termLabels{tt}), 'Interpreter','none');

        legLabels = arrayfun(@(s) sprintf('Subj %d', subj_list(s)), 1:nSubj, 'UniformOutput', false);
        legend([hLine; hMean], [legLabels, {'Mean (black)'}], 'Location','best');
    end
end


%% 10) Make ONE big multi-panel figure (term x model) --------------------
% Compatibility: subplot only; global legend via annotation

assert(exist('Sel','var')==1 && numel(Sel)>0, ...
    'Sel 不存在/为空：先运行 Section 9 PASS 1 生成 Sel。');

% -------- settings --------
fontPanel = 8;

lw_sub   = 0.55;
lw_mean  = 0.90;
alphaSub = 0.14;
alphaMean= 0.08;

nCols = numel(Sel);
nTermsEach = cellfun(@numel, {Sel.termLabels});
nRows = max(nTermsEach);

x = t_norm(:)';

% -------- figure size --------
figW = 1200;
figH = max(850, 190*nRows);
fig = figure('Color','w','Position',[80 80 figW figH]);

% -------- panels --------
for r = 1:nRows
    for c = 1:nCols

        ax = subplot(nRows, nCols, (r-1)*nCols + c);
        hold(ax,'on'); grid(ax,'on');

        mName = Sel(c).mName;

        % blank if this model doesn't have this term
        if r > numel(Sel(c).termLabels)
            axis(ax,'off');
            continue;
        end

        termLabel = Sel(c).termLabels{r};
        beta_sub  = Sel(c).beta_sub(:,:,r);   % nSubj x K
        se_sub    = Sel(c).se_sub(:,:,r);

        % subject lines + bands
        for iSub = 1:nSubj
            yv = squeeze(beta_sub(iSub,:));
            ev = squeeze(se_sub(iSub,:));

            ok = ~isnan(yv) & ~isnan(ev);
            if sum(ok) >= 2
                xx = x(ok); yy = yv(ok); ee = ev(ok);
                fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(iSub,:), ...
                    'EdgeColor','none', 'FaceAlpha', alphaSub, 'HandleVisibility','off');
            end

            plot(ax, x, yv, '-', 'Color', colSub(iSub,:), ...
                'LineWidth', lw_sub, 'HandleVisibility','off');
        end

        % mean ± SEM
        yMean = mean(beta_sub, 1, 'omitnan');
        ySEM  = std(beta_sub, 0, 1, 'omitnan') ./ sqrt(nSubj);

        okm = ~isnan(yMean) & ~isnan(ySEM);
        if sum(okm) >= 2
            xx = x(okm); ym = yMean(okm); es = ySEM(okm);
            fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                'EdgeColor','none', 'FaceAlpha', alphaMean, 'HandleVisibility','off');
        end
        plot(ax, x, yMean, 'k-', 'LineWidth', lw_mean, 'HandleVisibility','off');

        % axes styling
        yline(ax, 0, 'k--', 'HandleVisibility','off');
        xlim(ax, [0 1]);
        ylim(ax, yLimGlobal);
        xticks(ax, 0:0.2:1);
        set(ax, 'FontSize', fontPanel);

        % titles/labels
        if r == 1
            title(ax, mName, 'Interpreter','none');
        end
        if c == 1
            ylabel(ax, termLabel, 'Interpreter','none');
        end
        if r < nRows
            set(ax,'XTickLabel',[]);
        else
            xlabel(ax,'Normalized time (0–1)');
        end
    end
end

% -------- Global legend annotation --------
legendTxt = sprintf(['Blue: Subj 1\nRed: Subj 2\nYellow: Subj 3\nBlack: Mean']);
annotation(fig, 'textbox', [0.012 0.83 0.10 0.14], ...
    'String', legendTxt, 'FitBoxToText','on', ...
    'BackgroundColor','white', 'EdgeColor',[0.2 0.2 0.2], ...
    'FontSize', 9);

% -------- Export PDF (vector) --------
outName = 'BigFigure_TermByModel_NoCorr.pdf';

set(fig,'Renderer','painters');
set(fig,'PaperUnits','inches');

pos = get(fig,'Position');
dpi = get(0,'ScreenPixelsPerInch');
w_in = pos(3)/dpi;
h_in = pos(4)/dpi;

set(fig,'PaperSize',[w_in h_in]);
set(fig,'PaperPosition',[0 0 w_in h_in]);

print(fig, outName, '-dpdf', '-painters');

fprintf('Saved: %s\n', outName);
