%% H1H2_subjectDivergence_test.m
% Purpose:
%   Test whether "interaction-term divergence" across subjects can be explained
%   by subject-specific main effects only (H1) vs requiring subject-specific
%   interaction weights (H2), per time bin.
%
% H1:
%   conf ~ base terms + (subject intercept shifts) + (subject-specific main-effect slopes)
%   interaction terms are SHARED across subjects
%
% H2:
%   H1 + (subject-specific interaction slopes) [optionally gated to late window]
%
% Output:
%   - per-bin AIC/BIC for H1 and H2
%   - DeltaBIC(t) = BIC(H2) - BIC(H1)  (negative => H2 better => need interaction divergence)
%   - plots + .mat save
%
% NOTE:
%   This script recomputes:
%     - RPF predicted performance p_perf_all
%     - residual volatility resVol_time from motion energy
%     - predictors perf/corr/rt
%   using your current pipeline logic.
%
% Author: ChatGPT (tailored to your current code structure)
% -------------------------------------------------------------------------

clear; clc;

%% ===================== USER SETTINGS =====================
% --- paths ---
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/Palamedes1'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/toolbox/boundedline-pkg-master'));
RPF_check_toolboxes;

data_path = '/Users/wuyuzheng/Documents/MATLAB/projects/volatility/all_with_me.mat';

% --- confidence threshold ---
CONF_TH = 0.5;

% --- motion energy volatility window ---
winLen_ME = 10;
tol_ME    = 1e-12;

% --- resVol bins ---
nBins  = 40;

% --- minimum trials per bin ---
minN = 50;

% --- which subjects expected? (baseline is subject 1; S2/S3 dummies for 2/3) ---
% If your subjIDs are not 1/2/3, the script will remap to 1..nSubj.

% --- choose which base model family to test (must be consistent with how you made terms) ---
% We'll define the same NEW family names you used:
%   2-way: PxC, PxV, PxR, VxC, CxR, RxV
%   3-way: PxVxC, PxCxR, PxVxR, VxCxR
%   4-way: PxVxCxR
%
% Choose which base model(s) to test:
%   - "bestOnly": pick the best among M0/M1...M9 by mean BIC across bins (base formula WITHOUT subj terms)
%   - "top4": pick top4 by mean BIC across bins
%   - "manual": specify manual list below
MODEL_PICK_MODE = "top4";   % "bestOnly" | "top4" | "manual"
MANUAL_MODELS   = ["M7_all2","M8_all2_all3","M9_full"]; % used only if manual

% --- H2: which interaction terms to allow subject-specific slopes for? ---
% Put names from:
%   twoWayNames  = ["PxC","PxV","PxR","VxC","CxR","RxV"];
%   threeWayNames= ["PxVxC","PxCxR","PxVxR","VxCxR"];
%   fourWayNames = "PxVxCxR";
H2_INTERACTIONS = ["VxC"];   % e.g., ["VxC"] to test corr×vol divergence specifically
% You can add more, e.g. ["VxC","PxVxR","PxVxCxR"]

% --- late gating for interaction subject-specific effects ---
USE_LATE_GATING = true;
LATE_TAU = 0.60;      % normalized time threshold for gating (>= tau = 1)
% ----------------------------------------------------------

OUT_TAG = datestr(now,'yyyymmdd_HHMMSS');
OUT_MAT = sprintf('H1H2_results_%s.mat', OUT_TAG);
OUT_PNG = sprintf('H1H2_DeltaBIC_%s.png', OUT_TAG);

%% ===================== 1) LOAD DATA =====================
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

valid = ~isnan(coh_all) & ~isnan(resp_all) & ~isnan(correct_all) & ...
        ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all);

coh           = coh_all(valid);
resp          = resp_all(valid);
Correct       = correct_all(valid);
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID_raw    = subjID_all(valid);
motion_energy = ME_cell_all(valid);
rt            = rt_all(valid);

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% Binary confidence
Conf = double(confCont >= CONF_TH);

%% ===================== 2) REMAP SUBJECT IDS TO 1..nSubj =====================
subj_list_raw = unique(subjID_raw(:))';
nSubj = numel(subj_list_raw);
subjID = nan(size(subjID_raw));

for i = 1:nSubj
    subjID(subjID_raw == subj_list_raw(i)) = i;
end

fprintf('Subjects found: %d (remapped to 1..%d)\n', nSubj, nSubj);
disp(table(subj_list_raw(:), (1:nSubj)', 'VariableNames', {'origID','newID'}));

if nSubj < 2
    error('Need at least 2 subjects for H1/H2 test.');
end

%% ===================== 3) MAP VOLATILITY CONDITION FOR RPF =====================
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;

%% ===================== 4) RPF -> predicted performance p_perf_all =====================
p_perf_all = nan(nTrials, 1);

for iSub = 1:nSubj
    thisSub = iSub;
    fprintf('\n=== Running RPF for subject %d ===\n', thisSub);

    idxS = (subjID == thisSub);

    coh_s     = coh(idxS);
    resp_s    = resp(idxS);
    correct_s = Correct(idxS);
    conf_s    = confCont(idxS);
    cond_s    = cond(idxS);

    if isempty(coh_s), continue; end
    nTr = numel(coh_s);

    resp01 = resp_s - 1; % 1->0, 2->1

    stim01 = resp01;
    wrong_idx = (correct_s == 0);
    stim01(wrong_idx) = 1 - resp01(wrong_idx);

    conf_clip = min(max(conf_s,0),1);
    edges     = [0, 0.25, 0.5, 0.75, 1];
    rating_s  = discretize(conf_clip, edges, 'IncludedEdge', 'right');
    rating_s(isnan(rating_s)) = 4;

    trialData = struct();
    trialData.stimID    = stim01(:)';
    trialData.response  = resp01(:)';
    trialData.rating    = rating_s(:)';
    trialData.correct   = correct_s(:)';
    trialData.x         = coh_s(:)';
    trialData.condition = cond_s(:)';
    trialData.RT        = nan(1,nTr);

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

    p_perf_trial = nan(nTr,1);
    nCond = numel(F1.data);

    for c = 1:nCond
        mask_c = (cond_s == c);
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

%% ===================== 5) COMPUTE RESIDUAL VOLATILITY (motion energy) =====================
evidence_strength   = cell(nTrials, 1);
volatility_strength = cell(nTrials, 1);

for tr = 1:nTrials
    frames = motion_energy{tr};
    trace  = frames(:)';

    last_nz = find(abs(trace) > tol_ME, 1, 'last');
    if isempty(last_nz)
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);

    if nFrames < winLen_ME
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    nWin  = nFrames - winLen_ME + 1;
    m_win = nan(1, nWin);
    s_win = nan(1, nWin);

    for w = 1:nWin
        seg      = trace_eff(w : w + winLen_ME - 1);
        m_win(w) = mean(seg);
        s_win(w) = std(seg);
    end

    evidence_strength{tr}   = m_win;
    volatility_strength{tr} = s_win;
end

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

resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 3, continue; end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    resid = y_use - Xb*beta;

    tmp2 = nan(size(y));
    tmp2(mask_b) = resid;
    resVol_mat(:, b) = tmp2;
end

mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');
resVol_time = (resVol_mat - mu_all) ./ sd_all;

fprintf('Residual volatility matrix: %d trials x %d bins\n', size(resVol_time,1), size(resVol_time,2));

%% ===================== 6) BUILD PREDICTORS (perf/corr/rt) =====================
% perf: global z
eps0   = 1e-4;
p_clip = min(max(p_perf_all, eps0), 1-eps0);
Fp_all = (p_clip - mean(p_clip,'omitnan')) ./ std(p_clip,'omitnan');

% corr: center
Cz_all = Correct - mean(Correct,'omitnan');

% RT: log + global z
rt_eps  = 1e-6;
rt_ref  = log(rt + rt_eps);
RTz_all = (rt_ref - mean(rt_ref,'omitnan')) ./ std(rt_ref,'omitnan');

%% ===================== 7) DEFINE MODEL FAMILY (same as your NEW family) =====================
baseLabels = {'(Intercept)','perf','corr','vol','rt'};

twoWayNames  = ["PxC","PxV","PxR","VxC","CxR","RxV"];
threeWayNames= ["PxVxC","PxCxR","PxVxR","VxCxR"];
fourWayNames = "PxVxCxR";

modelNames = {};
modelSpec  = struct('use2',{},'use3',{},'use4',{});

% M0
idx = numel(modelNames)+1;
modelNames{idx}     = 'M0_base';
modelSpec(idx).use2 = false(1,6);
modelSpec(idx).use3 = false(1,4);
modelSpec(idx).use4 = false;

% M1
idx = numel(modelNames)+1;
modelNames{idx}     = 'M1_PC';
modelSpec(idx).use2 = false(1,6); modelSpec(idx).use2(1)=true;
modelSpec(idx).use3 = false(1,4);
modelSpec(idx).use4 = false;

% M2–M6 (same order you used)
oneAtATime = [5 6 2 3 4]; % CxR, RxV, PxV, PxR, VxC
for ii = 1:numel(oneAtATime)
    j = oneAtATime(ii);
    idx = numel(modelNames)+1;
    modelNames{idx} = sprintf('M%d_2way_%s', 1+ii, twoWayNames(j));
    modelSpec(idx).use2 = false(1,6); modelSpec(idx).use2(j)=true;
    modelSpec(idx).use3 = false(1,4);
    modelSpec(idx).use4 = false;
end

% M7
idx = numel(modelNames)+1;
modelNames{idx}     = 'M7_all2';
modelSpec(idx).use2 = true(1,6);
modelSpec(idx).use3 = false(1,4);
modelSpec(idx).use4 = false;

% M8
idx = numel(modelNames)+1;
modelNames{idx}     = 'M8_all2_all3';
modelSpec(idx).use2 = true(1,6);
modelSpec(idx).use3 = true(1,4);
modelSpec(idx).use4 = false;

% M9
idx = numel(modelNames)+1;
modelNames{idx}     = 'M9_full';
modelSpec(idx).use2 = true(1,6);
modelSpec(idx).use3 = true(1,4);
modelSpec(idx).use4 = true;

nModels = numel(modelNames);

%% ===================== 8) FIRST PASS: pick base models by mean BIC across bins =====================
% Here we fit the base formula WITHOUT any subject-specific slopes (no S2/S3 etc.),
% just to decide which model(s) to test with H1/H2.

AIC0 = nan(nBins, nModels);
BIC0 = nan(nBins, nModels);
Nobs0= nan(nBins, nModels);

for m = 1:nModels
    fprintf('\n[Model selection pass] Fitting base %s ...\n', modelNames{m});

    for k = 1:nBins
        Vk = resVol_time(:,k);

        mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Cz_all) & ~isnan(Fp_all) & ~isnan(RTz_all) & ~isnan(subjID);
        if sum(mask) < minN, continue; end

        y = Conf(mask);
        P = Fp_all(mask);
        C = Cz_all(mask);

        % zscore vol within this bin (global across subjects)
        Vraw = Vk(mask);
        sv = std(Vraw);
        if sv < 1e-12, continue; end
        V = (Vraw - mean(Vraw)) ./ sv;

        R = RTz_all(mask);

        % interactions
        PxC = P.*C; PxV = P.*V; PxR = P.*R;
        VxC = V.*C; CxR = C.*R; RxV = R.*V;

        PxVxC = P.*V.*C;
        PxCxR = P.*C.*R;
        PxVxR = P.*V.*R;
        VxCxR = V.*C.*R;

        PxVxCxR = P.*V.*C.*R;

        T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, ...
            'VariableNames', {'conf','perf','corr','vol','rt', ...
                              'PxC','PxV','PxR','VxC','CxR','RxV', ...
                              'PxVxC','PxCxR','PxVxR','VxCxR', ...
                              'PxVxCxR'});

        f = "conf ~ perf + corr + vol + rt";
        for j = 1:6
            if modelSpec(m).use2(j), f = f + " + " + twoWayNames(j); end
        end
        for j = 1:4
            if modelSpec(m).use3(j), f = f + " + " + threeWayNames(j); end
        end
        if modelSpec(m).use4, f = f + " + " + fourWayNames; end

        try
            g = fitglm(T, f, 'Distribution','binomial', 'Link','logit');
        catch
            continue;
        end

        AIC0(k,m)  = g.ModelCriterion.AIC;
        BIC0(k,m)  = g.ModelCriterion.BIC;
        Nobs0(k,m) = sum(mask);
    end
end

meanBIC0 = mean(BIC0, 1, 'omitnan');
[~, ord] = sort(meanBIC0, 'ascend', 'MissingPlacement','last');

if MODEL_PICK_MODE == "bestOnly"
    pickIdx = ord(1);
elseif MODEL_PICK_MODE == "top4"
    pickIdx = ord(1:min(4,numel(ord)));
elseif MODEL_PICK_MODE == "manual"
    pickIdx = [];
    for i = 1:numel(MANUAL_MODELS)
        hit = find(strcmp(modelNames, MANUAL_MODELS(i)), 1, 'first');
        if ~isempty(hit), pickIdx(end+1) = hit; end %#ok<AGROW>
    end
    pickIdx = unique(pickIdx);
else
    error('Unknown MODEL_PICK_MODE');
end

pickNames = string(modelNames(pickIdx));
fprintf('\n=== Models selected for H1/H2 test (%s) ===\n', MODEL_PICK_MODE);
disp(pickNames(:));

%% ===================== 9) H1/H2 TEST PER BIN FOR SELECTED MODELS =====================
% We build pooled regression per bin including subject dummies and subject-specific slopes.

% Prepare subject dummies: baseline is subject 1, then S2..Sn
S = cell(1,nSubj);
for s = 2:nSubj
    S{s} = double(subjID == s);
end

% late gating vector per bin (0/1)
gLate = double(t_norm(:) >= LATE_TAU); % nBins x 1

Results = struct();
Ridx = 0;

for mm = 1:numel(pickIdx)
    m = pickIdx(mm);
    Ridx = Ridx + 1;

    mName = modelNames{m};
    fprintf('\n=== H1/H2 for %s ===\n', mName);

    % which interaction terms exist in this base model?
    baseIntTerms = strings(0,1);

    for j = 1:6
        if modelSpec(m).use2(j), baseIntTerms(end+1,1) = twoWayNames(j); end %#ok<AGROW>
    end
    for j = 1:4
        if modelSpec(m).use3(j), baseIntTerms(end+1,1) = threeWayNames(j); end %#ok<AGROW>
    end
    if modelSpec(m).use4
        baseIntTerms(end+1,1) = fourWayNames; %#ok<AGROW>
    end

    % choose which of H2_INTERACTIONS are actually in this model
    % H2 includes ALL interactions present in this model
    H2_terms = baseIntTerms(:);

    if isempty(H2_terms)
        warning('No requested H2 interaction terms are present in %s. H2 will equal H1.', mName);
    else
        fprintf('H2 will add subject-specific slopes for: %s\n', strjoin(H2_terms, ', '));
        if USE_LATE_GATING
            fprintf('Late-gating ON at tau = %.2f\n', LATE_TAU);
        else
            fprintf('Late-gating OFF (all time bins)\n');
        end
    end

    AIC_H1 = nan(nBins,1);
    BIC_H1 = nan(nBins,1);
    AIC_H2 = nan(nBins,1);
    BIC_H2 = nan(nBins,1);
    Nobs   = nan(nBins,1);

    for k = 1:nBins
        Vk = resVol_time(:,k);

        mask = ~isnan(Vk) & ~isnan(Conf) & ~isnan(Cz_all) & ~isnan(Fp_all) & ~isnan(RTz_all) & ~isnan(subjID);
        if sum(mask) < minN, continue; end
        minPerSub = 15;   % 推荐 10–20，你现在数据我建议 15

        okSub = true;
        for s = 1:nSubj
            if sum(mask & subjID==s) < minPerSub
                okSub = false;
                break;
            end
        end

        if ~okSub
            % 这个 bin 某个被试 trial 太少
            % H1 / H2 都不要 fit
            continue;
        end

        y    = Conf(mask);
        P    = Fp_all(mask);
        C    = Cz_all(mask);

        Vraw = Vk(mask);
        sv = std(Vraw);
        if sv < 1e-12, continue; end
        V = (Vraw - mean(Vraw)) ./ sv;

        R = RTz_all(mask);

        sid = subjID(mask);

        % subject dummy columns in this masked set
        % Create S2..Sn for current mask
        Sd = [];
        Sd_names = strings(0,1);
        for s = 2:nSubj
            col = double(sid == s);
            Sd = [Sd, col]; %#ok<AGROW>
            Sd_names(end+1,1) = "S"+string(s); %#ok<AGROW>
        end

        % interactions
        PxC = P.*C; PxV = P.*V; PxR = P.*R;
        VxC = V.*C; CxR = C.*R; RxV = R.*V;

        PxVxC = P.*V.*C;
        PxCxR = P.*C.*R;
        PxVxR = P.*V.*R;
        VxCxR = V.*C.*R;

        PxVxCxR = P.*V.*C.*R;

        % base table (always includes all potential interaction columns; formula chooses subset)
        T = table(y,P,C,V,R, PxC,PxV,PxR,VxC,CxR,RxV, PxVxC,PxCxR,PxVxR,VxCxR, PxVxCxR, ...
            'VariableNames', {'conf','perf','corr','vol','rt', ...
                              'PxC','PxV','PxR','VxC','CxR','RxV', ...
                              'PxVxC','PxCxR','PxVxR','VxCxR', ...
                              'PxVxCxR'});

        % add subject dummy columns to table
        for si = 1:numel(Sd_names)
            T.(Sd_names(si)) = Sd(:,si);
        end

        % ---------- H1: add subject-specific main-effect slopes ----------
        % We'll add columns like:
        %   S2_perf = S2.*perf, S2_corr, S2_vol, S2_rt  (for each subject dummy)
        % and include them in the formula.
        mainNames = ["perf","corr","vol","rt"];

        subjMainCols = strings(0,1);
        for si = 1:numel(Sd_names)
            Sname = Sd_names(si);

            for mn = mainNames
                newNm = Sname + "_" + mn;
                T.(newNm) = T.(mn) .* T.(Sname);
                subjMainCols(end+1,1) = newNm; %#ok<AGROW>
            end
        end

        % Base formula (your chosen model terms)
        f_base = "conf ~ perf + corr + vol + rt";
        for j = 1:6
            if modelSpec(m).use2(j), f_base = f_base + " + " + twoWayNames(j); end
        end
        for j = 1:4
            if modelSpec(m).use3(j), f_base = f_base + " + " + threeWayNames(j); end
        end
        if modelSpec(m).use4, f_base = f_base + " + " + fourWayNames; end

        % Subject intercept shifts: + S2 + S3 + ...
        f_subjInt = f_base;
        for si = 1:numel(Sd_names)
            f_subjInt = f_subjInt + " + " + Sd_names(si);
        end

        % H1 formula: subject intercepts + subject-specific main-effect slopes
        f_H1 = f_subjInt;
        for iCol = 1:numel(subjMainCols)
            f_H1 = f_H1 + " + " + subjMainCols(iCol);
        end

        % ---------- H2: add subject-specific interaction slopes (optionally gated) ----------
        subjIntCols = strings(0,1);

        if ~isempty(H2_terms)
            for si = 1:numel(Sd_names)
                Sname = Sd_names(si);

                for it = 1:numel(H2_terms)
                    itName = H2_terms(it);
                    newNm  = Sname + "_" + itName;

                    % build column
                    if USE_LATE_GATING
                        gate = gLate(k);   % scalar 0/1 for this bin
                        col  = gate .* T.(itName) .* T.(Sname);
                    else
                        col  = T.(itName) .* T.(Sname);
                    end

                    % ---- robustness: only keep if it has variance and not all-zero ----
                    if all(col==0) || std(col) < 1e-12 || all(isnan(col))
                        continue;
                    end

                    T.(newNm) = col;
                    subjIntCols(end+1,1) = newNm; %#ok<AGROW>
                end
            end
        end


        f_H2 = f_H1;
        for iCol = 1:numel(subjIntCols)
            f_H2 = f_H2 + " + " + subjIntCols(iCol);
        end

        % ---------- Fit ----------
        try
            g1 = fitglm(T, f_H1, 'Distribution','binomial', 'Link','logit');
        catch
            continue;
        end

        try
            g2 = fitglm(T, f_H2, 'Distribution','binomial', 'Link','logit');
        catch
            % if H2 fails, keep H1 only
            g2 = [];
        end

        AIC_H1(k) = g1.ModelCriterion.AIC;
        BIC_H1(k) = g1.ModelCriterion.BIC;
        Nobs(k)   = sum(mask);

        if ~isempty(g2)
            AIC_H2(k) = g2.ModelCriterion.AIC;
            BIC_H2(k) = g2.ModelCriterion.BIC;
        else
            AIC_H2(k) = NaN;
            BIC_H2(k) = NaN;
        end
    end

    Results(Ridx).modelName = mName;
    Results(Ridx).H2_terms  = H2_terms;
    Results(Ridx).AIC_H1    = AIC_H1;
    Results(Ridx).BIC_H1    = BIC_H1;
    Results(Ridx).AIC_H2    = AIC_H2;
    Results(Ridx).BIC_H2    = BIC_H2;
    Results(Ridx).Nobs      = Nobs;
    Results(Ridx).DeltaBIC  = BIC_H2 - BIC_H1;
    Results(Ridx).DeltaAIC  = AIC_H2 - AIC_H1;
end

%% ===================== 10) PLOT: DELTA BIC ACROSS TIME =====================
fig = figure('Color','w','Units','points','Position',[80 80 1050 420]);
hold on; grid on; box off;

for i = 1:numel(Results)
    plot(t_norm, Results(i).DeltaBIC, 'LineWidth', 1.6);
end

yline(0,'k--','LineWidth',1);
xline(LATE_TAU,'k:','LineWidth',1);

xlabel('Normalized time (0–1)');
ylabel('\DeltaBIC = BIC(H2) - BIC(H1)');
titleStr = "H1 vs H2 interaction-divergence test";
if USE_LATE_GATING
    titleStr = titleStr + sprintf(' (late-gating @ %.2f)', LATE_TAU);
end
title(titleStr);

leg = strings(1,numel(Results));
for i = 1:numel(Results)
    leg(i) = string(Results(i).modelName);
end
legend(leg, 'Location','best');

% interpretive text
text(0.02, 0.05, ...
    'Negative \DeltaBIC => H2 better => needs subject-specific interaction weight', ...
    'Units','normalized', 'FontSize',10);

set(fig,'Renderer','painters');
print(fig, OUT_PNG, '-dpng', '-r200');
fprintf('✓ Saved plot: %s\n', OUT_PNG);

%% ===================== 11) SAVE RESULTS =====================
save(OUT_MAT, ...
    'Results','t_norm','MODEL_PICK_MODE','pickNames','H2_INTERACTIONS', ...
    'USE_LATE_GATING','LATE_TAU','meanBIC0','modelNames','AIC0','BIC0','Nobs0');

fprintf('✓ Saved results: %s\n', OUT_MAT);

fprintf('\nDONE.\n');

