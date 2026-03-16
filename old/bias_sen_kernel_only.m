%% resVol_RPF_betas_only.m
% Output only:
%   - per-subject b_V(t) (bias, logit) with error bars + per-subject significance
%   - per-subject b_{VxC}(t) (sens, logit) with error bars + per-subject significance
%
% Model (per subject, per time bin):
%   Conf ~ f(p_j) + Correct + V + V*Correct
%
% b_V(t)  = coefficient on V_z
% b_VxC(t)= coefficient on V_z * Correct

clear; clc;

%% 0) Add toolboxes ------------------------------------------------------
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/Palamedes1'));
RPF_check_toolboxes;

%% 1) Load data ----------------------------------------------------------
data_path = '/Users/wuyuzheng/Documents/MATLAB/all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh_all       = allStruct.rdm1_coh(:);
resp_all      = allStruct.req_resp(:);        % 1=right,2=left
correct_all   = allStruct.correct(:);         % 1/0
confCont_all  = allStruct.confidence(:);      % 0-1
vol_all       = allStruct.rdm1_coh_std(:);
subjID_all    = allStruct.group(:);           % 1/2/3
ME_cell_all   = allStruct.motion_energy;      % cell

valid = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
        ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all);

coh           = coh_all(valid);
resp          = resp_all(valid);
Correct       = correct_all(valid);
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = ME_cell_all(valid);

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% binary confidence
th   = 0.5;
Conf = double(confCont >= th);

%% 2) Volatility condition index (for RPF) -------------------------------
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1; % low
cond(vol == max(vol_levels)) = 2; % high

%% 3) RPF -> per-trial predicted performance p_perf_all ------------------
subj_list   = unique(subjID);
nSubj       = numel(subj_list);
p_perf_all  = nan(nTrials, 1);

for iSub = 1:nSubj
    thisSub = subj_list(iSub);
    fprintf('\n=============================\n');
    fprintf('Running RPF for subject %d\n', thisSub);
    fprintf('=============================\n');

    idxS = (subjID == thisSub);

    coh_s     = coh(idxS);
    resp_s    = resp(idxS);
    correct_s = Correct(idxS);
    conf_s    = confCont(idxS);
    cond_s    = cond(idxS);

    if isempty(coh_s)
        warning('Subject %d has no trials. Skipping.', thisSub);
        continue;
    end
    nTr = numel(coh_s);

    % resp 1/2 -> 0/1
    resp01 = resp_s - 1;

    % stim01: correct -> stim=resp; wrong -> stim=1-resp
    stim01 = resp01;
    stim01(correct_s == 0) = 1 - resp01(correct_s == 0);

    % confidence 0-1 -> rating 1..4
    conf_clip = min(max(conf_s,0),1);
    edges     = [0, 0.25, 0.5, 0.75, 1];
    rating_s  = discretize(conf_clip, edges, 'IncludedEdge','right');
    rating_s(isnan(rating_s)) = 4;

    trialData = struct();
    trialData.stimID    = stim01(:)';
    trialData.response  = resp01(:)';
    trialData.rating    = rating_s(:)';
    trialData.correct   = correct_s(:)';
    trialData.x         = coh_s(:)';
    trialData.condition = cond_s(:)';
    trialData.RT        = nan(1,nTr);

    % F1 performance PF
    F1 = struct();
    F1.info.DV                     = 'd''';
    F1.info.PF                     = @RPF_scaled_Weibull;
    F1.info.padCells               = 1;
    F1.info.set_P_max_to_d_pad_max = 1;
    F1.info.x_min                  = 0;
    F1.info.x_max                  = 1;
    F1.info.x_label                = 'coherence';
    F1.info.cond_labels            = {'low volatility','high volatility'};
    F1 = RPF_get_F(F1.info, trialData);

    % per-trial p(correct) from fitted d'(coh)
    p_perf_trial = nan(nTr,1);
    nCond = numel(F1.data);

    for c = 1:nCond
        mask_c = (cond_s == c);
        if ~any(mask_c), continue; end

        coh_c  = coh_s(mask_c);
        x_grid = F1.data(c).x(:);
        d_grid = F1.data(c).P(:);

        [~, loc] = ismember(coh_c, x_grid);
        if any(loc == 0)
            needInterp = (loc == 0);
            d_interp   = interp1(x_grid, d_grid, coh_c(needInterp), 'linear','extrap');
            loc(needInterp) = numel(d_grid) + 1;
            d_grid_ext = [d_grid; d_interp(:)];
            d_pred = d_grid_ext(loc);
        else
            d_pred = d_grid(loc);
        end

        p_corr = normcdf(d_pred ./ sqrt(2)); % 2AFC
        p_perf_trial(mask_c) = p_corr;
    end

    p_perf_all(idxS) = p_perf_trial;
end

fprintf('Finished RPF. Valid p_perf proportion: %.3f\n', mean(~isnan(p_perf_all)));

%% 4) Motion energy -> residual volatility matrix resVol_time (N x K) ----
winLen = 10;
tol    = 1e-12;

evidence_strength   = cell(nTrials,1);
volatility_strength = cell(nTrials,1);

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
    m_win = nan(1,nWin);
    s_win = nan(1,nWin);

    for w = 1:nWin
        seg      = trace_eff(w:w+winLen-1);
        m_win(w) = mean(seg);
        s_win(w) = std(seg);
    end

    evidence_strength{tr}   = m_win;
    volatility_strength{tr} = s_win;
end

nBins  = 40;
t_norm = linspace(0,1,nBins);

MEAN_norm = nan(nTrials,nBins);
STD_norm  = nan(nTrials,nBins);

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};
    if isempty(mu_tr) || isempty(sd_tr), continue; end

    nWin_tr = min(numel(mu_tr), numel(sd_tr));
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    t_orig = linspace(0,1,nWin_tr);
    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

resVol_mat = nan(size(STD_norm));
for b = 1:nBins
    y  = STD_norm(:,b);
    x1 = abs(MEAN_norm(:,b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 10, continue; end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    resid = y_use - Xb*beta;

    tmp = nan(size(y));
    tmp(mask_b) = resid;
    resVol_mat(:,b) = tmp;
end

% global z-score
mu_all = mean(resVol_mat(:),'omitnan');
sd_all = std(resVol_mat(:),'omitnan');
resVol_time = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d time bins.\n', size(resVol_time,1), size(resVol_time,2));

%% 5) performance predictor f(p_j) --------------------------------------
eps0   = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);
f_perf = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

%% 6) Per-subject logistic betas (+ SE + p) ------------------------------
uniqSubj = unique(subjID);
nSubj    = numel(uniqSubj);
K        = numel(t_norm);

beta_V_subj   = nan(nSubj, K);
beta_VxC_subj = nan(nSubj, K);

se_V_subj     = nan(nSubj, K);
se_VxC_subj   = nan(nSubj, K);

p_V_subj      = nan(nSubj, K);
p_VxC_subj    = nan(nSubj, K);

alpha = 0.05;
useFDR_withinSubj = false;  % 想做每个subject的bin-wise FDR就改true

for s = 1:nSubj
    thisSub = uniqSubj(s);
    fprintf('Fitting betas: Subj %d\n', thisSub);

    idxS = (subjID == thisSub);

    Conf_s    = Conf(idxS);
    Correct_s = Correct(idxS);
    f_perf_s  = f_perf(idxS);
    Vmat_s    = resVol_time(idxS,:);

    for k = 1:K
        Vk = Vmat_s(:,k);

        mask = ~isnan(Vk) & ~isnan(Conf_s) & ~isnan(Correct_s) & ~isnan(f_perf_s);
        if sum(mask) < 30
            continue;
        end

        y  = Conf_s(mask);
        C  = Correct_s(mask);
        Fp = f_perf_s(mask);
        V  = Vk(mask);

        % z-score V within (subject,timebin)
        if std(V) == 0 || isnan(std(V))
            continue;
        end
        V_z = (V - mean(V)) ./ std(V);
        VxC = V_z .* C;

        X = [Fp, C, V_z, VxC]; % glmfit adds intercept
        [b, ~, stats] = glmfit(X, y, 'binomial', 'link','logit');

        % b: [b0 bP bC bV bVC]
        beta_V_subj(s,k)   = b(4);
        beta_VxC_subj(s,k) = b(5);

        % SE & p from glmfit stats
        se_V_subj(s,k)     = stats.se(4);
        se_VxC_subj(s,k)   = stats.se(5);

        p_V_subj(s,k)      = stats.p(4);
        p_VxC_subj(s,k)    = stats.p(5);
    end

    % optional: within-subject FDR across bins
    if useFDR_withinSubj
        p_V_subj(s,:)   = fdr_bh(p_V_subj(s,:));
        p_VxC_subj(s,:) = fdr_bh(p_VxC_subj(s,:));
    end
end

%% 7) Plot only: BETAS (3 lines) + error bars + per-subject significance -
colors = lines(nSubj);

% 用于星号“离 errorbar 顶端”的间距
range_bV  = range(beta_V_subj(:),   'omitnan');
if isnan(range_bV) || range_bV == 0, range_bV = 0.05; end

range_bVC = range(beta_VxC_subj(:), 'omitnan');
if isnan(range_bVC) || range_bVC == 0, range_bVC = 0.05; end

dx_bV   = 0.010;
gap_bV  = 0.03 * range_bV;

dx_bVC  = 0.010;
gap_bVC = 0.03 * range_bVC;

starSize = 6;
starLW   = 1.0;

cap = 8;
lw  = 1.6;

% ================= Figure 1: b_V(t) =================
figure; hold on;
hBias = gobjects(nSubj,1);

for s = 1:nSubj
    hBias(s) = errorbar(t_norm, beta_V_subj(s,:), se_V_subj(s,:), '-o', ...
        'Color', colors(s,:), 'LineWidth', lw, 'CapSize', cap);
    hBias(s).DisplayName = sprintf('Subj %d', uniqSubj(s));

    sigK = find(p_V_subj(s,:) < alpha);
    for ii = 1:numel(sigK)
        k   = sigK(ii);
        yk  = beta_V_subj(s,k);
        sek = se_V_subj(s,k);

        if ~isnan(yk) && ~isnan(sek)
            side = sign(yk); if side==0, side=1; end
            xStar = t_norm(k) + dx_bV;
            yStar = yk + side*sek + side*gap_bV;

            plot(xStar, yStar, '*', ...
                'Color', colors(s,:), ...
                'MarkerSize', starSize, ...
                'LineWidth', starLW, ...
                'HandleVisibility', 'off');
        end
    end
end

yline(0,'k--'); grid on;
xlabel('Normalized time');
ylabel('b_V(t) (logit)');

ttl = 'Per-subject b_V(t) (bias, logit) + SE + per-subject significance';
if useFDR_withinSubj
    ttl = [ttl ' (stars: q<0.05 within-subj)'];
else
    ttl = [ttl ' (stars: p<0.05 within-subj)'];
end
title(ttl);
legend(hBias, 'Location','best');
xlim([min(t_norm), max(t_norm)+0.03]);

% ============== Figure 2: b_{V×C}(t) ==============
figure; hold on;
hSens = gobjects(nSubj,1);

for s = 1:nSubj
    hSens(s) = errorbar(t_norm, beta_VxC_subj(s,:), se_VxC_subj(s,:), '-o', ...
        'Color', colors(s,:), 'LineWidth', lw, 'CapSize', cap);
    hSens(s).DisplayName = sprintf('Subj %d', uniqSubj(s));

    sigK = find(p_VxC_subj(s,:) < alpha);
    for ii = 1:numel(sigK)
        k   = sigK(ii);
        yk  = beta_VxC_subj(s,k);
        sek = se_VxC_subj(s,k);

        if ~isnan(yk) && ~isnan(sek)
            side = sign(yk); if side==0, side=1; end
            xStar = t_norm(k) + dx_bVC;
            yStar = yk + side*sek + side*gap_bVC;

            plot(xStar, yStar, '*', ...
                'Color', colors(s,:), ...
                'MarkerSize', starSize, ...
                'LineWidth', starLW, ...
                'HandleVisibility', 'off');
        end
    end
end

yline(0,'k--'); grid on;
xlabel('Normalized time');
ylabel('b_{V\times C}(t) (logit)');

ttl = 'Per-subject b_{V×C}(t) (sens, logit) + SE + per-subject significance';
if useFDR_withinSubj
    ttl = [ttl ' (stars: q<0.05 within-subj)'];
else
    ttl = [ttl ' (stars: p<0.05 within-subj)'];
end
title(ttl);
legend(hSens, 'Location','best');
xlim([min(t_norm), max(t_norm)+0.03]);

%% ---------------- Local function: BH FDR -------------------------------
function q = fdr_bh(p)
% Benjamini-Hochberg FDR (returns q-values). Works with NaNs.
    q = nan(size(p));
    p0 = p(:);

    idxValid = find(~isnan(p0));
    pv = p0(idxValid);
    [ps, order] = sort(pv, 'ascend');

    m = numel(ps);
    if m == 0, return; end

    qtmp = ps .* m ./ (1:m)';
    for i = m-1:-1:1
        qtmp(i) = min(qtmp(i), qtmp(i+1));
    end
    qtmp(qtmp>1) = 1;

    qv = nan(m,1);
    qv(order) = qtmp;

    q0 = nan(size(p0));
    q0(idxValid) = qv;

    q = reshape(q0, size(p));
end
